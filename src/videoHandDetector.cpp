//
//  videoHandDetector.cpp
//  Fingerprint
//
//  Created by Luis Valdivia on 12.05.26.
//
#include <initializer_list>
#include <ranges>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <array>
#include "torch/torch.h"
#include "torch/script.h"
#include <opencv2/geometry/2d.hpp>
#include <opencv2/ximgproc.hpp>
#include "videoHandDetector.hpp"
#include "tasks/cc/vision/hand_landmarker/hand_landmarker.h"
#include "framework/formats/image.h"
#include "fframework/formats/image_frame.h"
#include <nfiq2.hpp>

//#include "modules/videoio/include/opencv2/videoio.hpp"

static inline std::string outputDebugImg  = "/path/to/debugImages/frame_";

static inline std::string outputDebugLandmarker  = "/path/to/debugLandmarker/frame_";

//static inline std::string outputDebugLandmarker  = "/path/to/debugFingertip/frame_";

static inline std::string outputDebugFingertipPath = "/path/to/fingertips/frame_";

static inline std::string modelFreezenPath = "/path/to/model/swinir_traced_frozen.pt";

static constexpr std::array<std::pair<uint8_t, uint8_t>, 5> fingerPoints = {std::make_pair(4,2), std::make_pair(8,6), std::make_pair(12,10), std::make_pair(16,14), std::make_pair(20,18)};

static constexpr std::array<std::pair<uint8_t, uint8_t>, 4> fingerPoints2 = {std::make_pair(8,7), std::make_pair(12,11), std::make_pair(16,15), std::make_pair(20,19)};

static inline const cv::Vec3f cameraPosition(0.F, 0.F, -1.F);

static constexpr float rotatiionN = 90.F;

[[nodiscard]] static constexpr bool isThumbFacingCamera(const BoxKp3& handLandmarks, const float threshold=0.85F) noexcept {
	cv::Vec3f v2(handLandmarks.points[2].x, handLandmarks.points[2].y, handLandmarks.points[2].z);
	cv::Vec3f v3(handLandmarks.points[3].x, handLandmarks.points[3].y, handLandmarks.points[3].z);
	cv::Vec3f v4(handLandmarks.points[4].x, handLandmarks.points[4].y, handLandmarks.points[4].z);
	
	cv::Vec3f bottom = v3 - v2;
	cv::Vec3f top = v4 -v3;
	cv::Vec3f normal = bottom.cross(top);
	normal = normal / cv::norm(normal);
	const float dotPr = normal.dot(cameraPosition);
	return dotPr > threshold;
}

[[nodiscard]] static constexpr double getFingerScore(const cv::Point3f& firstPoint, const cv::Point3f& secondPoint, const double proportionOfHeight, const size_t height) noexcept {
	double score = std::hypot(firstPoint.x - secondPoint.x, firstPoint.y - secondPoint.y);
	score = score * (double)height;
	return score < proportionOfHeight ? 0.0 : score;
}

[[nodiscard]] static constexpr std::array<double, 5> calculateSuitability(const BoxKp3& handLandmarks, const double proportionOfHeight, const size_t height) noexcept {
	std::array<double, 5> fingerScores;
	size_t count {0};
	for(const auto& points : fingerPoints) {
		fingerScores[count] = getFingerScore(handLandmarks.points[points.first], handLandmarks.points[points.second], proportionOfHeight, height);
		count++;
	}
	return fingerScores;
}

[[nodiscard]] const std::optional<cv::Mat> getThumbRoi(const cv::Mat& frame, const BoxKp3& handLandmarks, const float padding=0.4F) {
	if(not isThumbFacingCamera(handLandmarks)) return std::nullopt;
	cv::Vec2f tip(handLandmarks.points[4].x * (float)frame.size().width, handLandmarks.points[4].y * (float)frame.size().height);
	cv::Vec2f dip(handLandmarks.points[2].x * (float)frame.size().width, handLandmarks.points[2].y * (float)frame.size().height);
	float thumbLenght = static_cast<float>(cv::norm(tip - dip));
	if(thumbLenght < 20.F) return std::nullopt; //magic number, this should be changed, for a proportion of the frame Height;
	const int targetSize = static_cast<int>(thumbLenght*(1.F+padding));
	//To rotate safely without clipping corners, we extract a larger "safe" region first
	const float sRadius = (float)targetSize * 0.75F;
	const cv::Vec2f centerPx = (tip+dip) / 2.F;
	const cv::Vec2f firstPos(centerPx[0]- sRadius, centerPx[1]-sRadius);
	const cv::Vec2f secondPos(centerPx[0] + sRadius, centerPx[1] + sRadius);
	const int padLeft = static_cast<int>(std::max(0.F, -firstPos[0]));
	const int padTop = static_cast<int>(std::max(0.F, -firstPos[1]));
	const int padRight = static_cast<int>(std::max(0.F, secondPos[0] - (float)frame.size().width));
	const int padBottom = static_cast<int>(std::max(0.F, secondPos[1] - (float)frame.size().height));
	const int validX1 = static_cast<int>(std::max(0.F, firstPos[0]));
	const int validY1 = static_cast<int>(std::max(0.F, firstPos[1]));
	const int validX2 = static_cast<int>(std::min((float)frame.size().width,secondPos[0]));
	const int validY2 = static_cast<int>(std::min((float)frame.size().height, secondPos[1]));
	cv::Mat rawCrop;
	frame(cv::Rect(validX1, validY1, validX2 - validX1, validY2 - validY1)).copyTo(rawCrop);
	cv::Mat_<float> safeCrop(padBottom - padTop, padRight - padLeft);
	cv::copyMakeBorder(rawCrop, safeCrop, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT);
	//Rotation
	//calculate current angle of the thumb
	const float dx = tip[0] - dip[0];
	const float dy = tip[1] - dip[1];
	const float angleDeg = (std::atan2f(dx, dy) * 180.F/std::numbers::pi_v<float>) + rotatiionN;
	const cv::Point2f centerCrop((float)safeCrop.cols/2.F, (float)safeCrop.rows/2.F);
	cv::Mat rotated = cv::getRotationMatrix2D(centerCrop, angleDeg, 1.0);
	cv::Mat rotatedCrop;
	cv::warpAffine(safeCrop, rotatedCrop, rotated, safeCrop.size());
	
	const int finalX1 = static_cast<int>(centerCrop.x - (float)targetSize / 2.F);
	const int finalY1 = static_cast<int>(centerCrop.y - (float)targetSize/2);
	const int finalX2 = finalX1 + targetSize;
	const int finalY2 = finalX2 + targetSize;
	cv::Mat finalRoi;
	rotatedCrop(cv::Rect(finalX1, finalY1, finalX2 - finalX1, finalY2-finalY1)).copyTo(finalRoi);
	return std::make_optional<cv::Mat>(finalRoi);
}

[[nodiscard]] const std::optional<cv::Mat> getAlignedThumbRoi(const cv::Mat& frame, const BoxKp3& handLandmarks) {
	cv::Point2f tip (handLandmarks.points[4].x * (float)frame.size().width, handLandmarks.points[4].y * (float)frame.size().height);
	cv::Point2f joint(handLandmarks.points[3].x * (float)frame.size().width, handLandmarks.points[3].y * (float)frame.size().height);
	//calculate angle
	const float dx = tip.x - joint.x;
	const float dy = tip.y - joint.y;
	const float angleDeg = (std::atan2(dy, dx) * 180.F/std::numbers::pi) + 90.F;
	
	const int finalSize = 64;
	//safe zone to accommodate the diagonal rotation of the square
	const int safeSize = static_cast<int>((float)finalSize * 1.5F);
	//extract a loose axis aligned bounding box
	cv::Rect looseRoi(static_cast<int>(tip.x - (float)safeSize/2.F), static_cast<int>(tip.y - (float)safeSize/2.F), safeSize, safeSize);
	
	cv::Rect frameBounds(0,0,frame.cols, frame.rows);
	cv::Rect intersection = looseRoi & frameBounds;
	//return null if the zone is clipped
	if(intersection.area() != looseRoi.area())
		return std::nullopt;
	
	cv::Mat looseCrop = frame(looseRoi);
	
	cv::Point2f centerOfRotation((float)safeSize / 2.F, (float) safeSize / 2.F);
	cv::Mat rotMat = cv::getRotationMatrix2D(centerOfRotation, angleDeg, 1.0);
	
	cv::Mat rotatedCrop;
	
	cv::warpAffine(looseCrop, rotatedCrop, rotMat, looseCrop.size(), cv::INTER_LINEAR);
	//take the final aligned center crop
	cv::Rect tightRoi((safeSize - finalSize) / 2, (safeSize - finalSize)/2, finalSize, finalSize);
	return std::make_optional(rotatedCrop(tightRoi).clone());
}

[[nodiscard]] const std::vector<Candidates> getFingertipRoi(const cv::Mat& frame, const BoxKp3& handLandmarks, const size_t frameNr, const float padding = 0.2F) noexcept {
	std::vector<Candidates> fingertips;
	cv::Point3f tip;
	cv::Point3f dip;
	size_t fingerCounter {1};
	std::string fileName = outputDebugFingertipPath;
	fileName += std::to_string(frameNr);
	if(const auto thumbRoi = getThumbRoi(frame, handLandmarks); thumbRoi.has_value()) {
		fileName += "_0.png";
		fingertips.emplace_back(Candidates{fileName, 0, thumbRoi.value()});
	}
	for(const auto& points : fingerPoints2) {
		cv::Mat roi;
		tip = handLandmarks.points[points.first] * frame.size().width;
		dip = handLandmarks.points[points.second] * frame.size().height;
		if(const double score = getFingerScore(tip, dip, frame.size().height*0.14, frame.size().height); score <= 0.0) {
			std::cout<<"Frame:"<<frameNr<<", finger:"<<fingerCounter<<", score:"<<score<<"\n";
			fingerCounter++;
			continue;
		}
		tip.x *= (float)frame.size().width;
		tip.y *= (float)frame.size().height;
		dip.x *= (float)frame.size().width;
		dip.y *= (float)frame.size().height;
		float fingerLength = std::hypotf(tip.x - dip.x, tip.y - dip.y);
		float boxSize = fingerLength*(1+padding*2);
		float x1 = std::max(0.F, tip.x - std::floor(boxSize/2.F));
		float y1 = std::max(0.F, tip.y - std::floor(boxSize/2.F));
		float x2 = std::min((float)frame.size().width, x1+boxSize);
		float y2 = std::min((float)frame.size().height, y1 + boxSize);
		float subMatrixWidth = x2 - x1;
		float subMatrixHeight = y2 - y1;
		frame(cv::Rect((int)x1, (int)y1, (int)subMatrixWidth, (int)subMatrixHeight)).copyTo(roi);
		fileName = outputDebugFingertipPath;
		fileName += std::to_string(frameNr) + "_" + std::to_string(fingerCounter) + ".png";
		fingertips.emplace_back(Candidates{fileName, fingerCounter, roi});
	}
	return fingertips;
}

/**
 * Aligns a sequence of ROIs to a base anchor frame using Dense Optical Flow.
 * * @param rois A std::vector of BGR cv::Mat images representing the extracted fingers.
 * @return A std::vector of perfectly aligned BGR images.
 */
/*
[[nodiscard]] static inline const std::vector<Candidates> alignFrames(const std::vector<Candidates>& rois) {
	if (rois.size() < 2)  return rois;
	std::vector<cv::Mat> aligned_rois;
	aligned_rois.reserve(rois.size());
	// Establish the "Anchor" Frame
	//cv::Mat ref_frame = rois[0].fingerCrop.clone();
	cv::Mat ref_gray;
	cv::cvtColor(rois[0].fingerCrop.clone(), ref_gray, cv::COLOR_BGR2GRAY);
	//rois[0].fingerCrop = ref_frame.clone();
	
	int rows = ref_gray.rows;
	int cols = ref_gray.cols;
	
	// Create the coordinate meshgrid representing destination pixel positions.
	cv::Mat grid_x(rows, cols, CV_32FC1);
	cv::Mat grid_y(rows, cols, CV_32FC1);
	for (int y = 0; y < rows; ++y) {
		float* x_ptr = grid_x.ptr<float>(y);
		float* y_ptr = grid_y.ptr<float>(y);
		for (int x = 0; x < cols; ++x) {
			x_ptr[x] = static_cast<float>(x);
			y_ptr[x] = static_cast<float>(y);
		}
	}
	
	// Loop through and warp the remaining frames
	for (size_t i = 1; i < rois.size(); ++i) {
		const cv::Mat& curr_frame = rois[i].fingerCrop;
		// Safety check to ensure all crops are the same size before processing
		if (curr_frame.size() != ref_gray.size())
			throw std::invalid_argument("All ROIs must be of the same dimensions.");
		cv::Mat curr_gray;
		cv::cvtColor(curr_frame, curr_gray, cv::COLOR_BGR2GRAY);
		//Compute Dense Optical Flow (Farneback)
		cv::Mat flow;
		cv::calcOpticalFlowFarneback(curr_gray, ref_gray, flow,
									 0.5,      // pyr_scale
									 3,        // levels
									 15,       // winsize (use 11 or 13 for non-thumb fingers if preferred)
									 3,        // iterations
									 5,        // poly_n
									 1.2,      // poly_sigma
									 0         // flags
									 );
		
		// Map the Flow Vectors to the Pixel Grid
		// We split it into two separate 1-channel matrices (dx and dy).
		std::vector<cv::Mat> flow_parts;
		cv::split(flow, flow_parts);
		
		cv::Mat map_x = grid_x + flow_parts[0];
		cv::Mat map_y = grid_y + flow_parts[1];
		
		//  Warp the Original BGR Frame
		cv::Mat warped_roi;
		cv::remap(curr_frame, warped_roi, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT,  cv::Scalar(0, 0, 0));
		warped_roi.copyTo(rois[i].fingerCrop);
//		rois[i].fingerCrop = warped_roi.clone();
	}
	return rois;
}
*/

static inline void drawHands(const cv::Mat& img, const std::vector<BoxKp2>& boxes, const size_t frameNumber) noexcept {
	std::string outputString = outputDebugImg;
	outputString += std::to_string(frameNumber) + ".png";
	for(const auto& box : boxes) {
		cv::rectangle(img, box.rect, cv::Scalar(255,0,0), 2);
		cv::putText(img, std::to_string(box.score), cv::Point((int)box.rect.x, (int)box.rect.y), 2, 1, cv::Scalar(0,255,0));
		for(size_t i = 0; i < box.points.size(); i++) {
			cv::putText(img, std::to_string(i), box.points[i], 2, 1, cv::Scalar(0,255,0));
			cv::circle(img, box.points[i], 2, cv::Scalar(255,0,255), 3);
		}
	}
	cv::imwrite(outputString, img);
}

static inline void drawHandLandmark(const cv::Mat& img, const std::vector<BoxKp3>& boxes, const size_t frameNumber) noexcept {
	std::string outputString = outputDebugLandmarker;
	outputString += std::to_string(frameNumber) + ".png";
	const int imgWidth = img.cols;
	const int imgHeight = img.rows;
	
	for(const auto& box : boxes) {
		cv::Point2f center = cv::Point2f(box.points[0].x * (float)imgWidth, box.points[0].y* (float)imgHeight);
		mpp::drawRotateRect(img, box.rect, box.radians, center);
		cv::putText(img, std::to_string(box.score), cv::Point((int)box.rect.x, (int)box.rect.y), 2, 1, cv::Scalar(0,255,0));
		for(size_t i = 0; i < box.points.size(); i++) {
			const auto p = cv::Point2f(box.points[i].x * (float)imgWidth, box.points[i].y * (float)imgHeight);
			cv::putText(img, std::to_string(i), p, 2, 1, cv::Scalar(0,255,0));			
			cv::circle(img, p, 2, cv::Scalar(255,0,255), 3);
		}
	}
	cv::imwrite(outputString, img);
}

void qualityAnalisisNFIQ(const cv::Mat& nfiqReady, const Candidates& rois) {
	// NFIQ 2 expects a flat array of 8-bit unsigned integers (grayscale)
	uint8_t* rawImageData = nfiqReady.data;
	uint32_t width = nfiqReady.cols;
	uint32_t height = nfiqReady.rows;
	uint8_t depth = 8; // 8-bit grayscale
	uint16_t ppi = 500; // Pixels Per Inch (DPI) - CRITICAL for NFIQ 2
	// Example using standard NFIQ 2.1 API structure:
	// You may need to wrap rawImageData in a std::vector<uint8_t> depending on your exact header
	std::vector<uint8_t> imgBuffer(rawImageData, rawImageData + (width * height));
	NFIQ2::ModelInfo modelInfoObj;
	NFIQ2::Algorithm model {};
	// Calculate all native quality measures.
	std::vector<std::shared_ptr<NFIQ2::QualityMeasures::Algorithm>>
	algorithms {};
	try {
		modelInfoObj = NFIQ2::ModelInfo("/usr/local/nfiq2/share/nist_plain_tir-ink.txt");
	}
	catch(...) {
		std::abort();
	}
	
	try {
		model = NFIQ2::Algorithm(modelInfoObj);
	}
	catch (...) {
		std::cerr << "Could not initialize model from model info file\n";
		std::abort();
	}
	
	NFIQ2::FingerprintImageData rawImage = NFIQ2::FingerprintImageData(imgBuffer.data(), imgBuffer.size(), width, height, 0, ppi);
	
	try {
		algorithms = NFIQ2::QualityMeasures::
		computeNativeQualityMeasureAlgorithms(rawImage);
	}
	catch(const NFIQ2::Exception& e) {
		std::cerr << "Error in calculating quality features: "
		<< e.what() << '\n';
		std::abort();
	}
	
	size_t nfiq2 {};
	
	try {
		nfiq2 = model.computeUnifiedQualityScore(algorithms);
	}
	catch (...) {
		std::cerr << "Error in calculating NFIQ 2 score\n";
		std::abort();
	}
	
	std::cout << "UnifiedQualityScore: " << nfiq2 << '\n';
	
	// Actionable Feedback
	std::vector<std::string> actionableIDs =
	NFIQ2::QualityMeasures::getActionableQualityFeedbackIDs();
	
	std::unordered_map<std::string, double> actionableQuality =
	NFIQ2::QualityMeasures::getActionableQualityFeedback(algorithms);
	
	for (const auto& actionableID : actionableIDs) {
		std::cout << actionableID << ": "
		<< actionableQuality.at(actionableID) << '\n';
	}
	
	// Quality Feature Values
	std::vector<std::string> featureIDs =
	NFIQ2::QualityMeasures::getNativeQualityMeasureIDs();
	
	std::unordered_map<std::string, double> qualityFeatures =
	NFIQ2::QualityMeasures::getNativeQualityMeasures(algorithms);
	
	for (const auto& featureID : featureIDs) {
		std::cout << featureID << ": " << qualityFeatures.at(featureID)
		<< '\n';
	}
}

[[nodiscard]] cv::Mat applyRidgeEnhancement2(const cv::Mat& hrImage) {
	cv::Mat gray;
	if (hrImage.channels() >= 3) cv::cvtColor(hrImage, gray, cv::COLOR_BGR2GRAY);
	else gray = hrImage.clone();
	
	// Aggressive Contrast Stretching
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(5.0, cv::Size(8,8));
	cv::Mat contrasted;
	clahe->apply(gray, contrasted);
	
	// Unsharp Mask (High-Pass Filter) to artificially boost hidden textures
	cv::Mat blur, sharp;
	cv::GaussianBlur(contrasted, blur, cv::Size(0, 0), 3);
	// This formula heavily amplifies the difference between the blurred and original image
	cv::addWeighted(contrasted, 2.5, blur, -1.5, 0, sharp);
	
	// Standard Adaptive Thresholding
	cv::Mat binary;
	cv::adaptiveThreshold(sharp, binary, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 15, 5);
	
	return binary;
}

[[nodiscard]] cv::Mat applyRidgeEnhancement(const cv::Mat& hrImage) {
	// Ensure 8-bit
	cv::Mat gray;
	if (hrImage.channels() >= 3) cv::cvtColor(hrImage, gray, cv::COLOR_BGR2GRAY);
	else gray = hrImage.clone();
	
	// Create a MASK to ignore the black background
	// Any pixel > 10 (very dark) is considered "finger"
	cv::Mat mask;
	cv::threshold(gray, mask, 10, 255, cv::THRESH_BINARY);
	
	// Apply CLAHE only inside the mask
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
	cv::Mat claheImg;
	clahe->apply(gray, claheImg);
	
	// Ridge Detection
	cv::Ptr<cv::ximgproc::RidgeDetectionFilter> ridgeFilter = cv::ximgproc::RidgeDetectionFilter::create();
	cv::Mat ridgeMap;
	ridgeFilter->getRidgeFilteredImage(claheImg, ridgeMap);
	
	// MASKING: Multiply the ridgeMap by the binary mask
	// This forces all background pixels to 0, removing the outer "garbage" lines
	cv::Mat maskedRidgeMap;
	ridgeMap.convertTo(maskedRidgeMap, CV_8UC1);
	maskedRidgeMap &= mask;
	
	// Final normalization and inversion
	cv::Mat enhanced;
	cv::normalize(maskedRidgeMap, enhanced, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	cv::bitwise_not(enhanced, enhanced);
	
	return enhanced;
}
//sharpening, edge detection clahe lassen
void classicalEnhanceLowResFirst(const Candidates& rois) {
	cv::Mat gray;
	if (rois.fingerCrop.channels() >= 3) cv::cvtColor(rois.fingerCrop, gray, cv::COLOR_BGR2GRAY);
	else gray = rois.fingerCrop.clone();
	
	//  Process at native 64x64 resolution where ridges are close together
	// A small 5x5 kernel perfectly matches this tight pixel scale
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
	cv::Mat ridges;
	cv::morphologyEx(gray, ridges, cv::MORPH_BLACKHAT, kernel);
	
	// Local contrast enhancement at micro-scale
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(4, 4));
	clahe->apply(ridges, ridges);
	
	//  Binarize and invert (Low thresholds work better on raw blurry values)
	cv::Mat binary;
	cv::threshold(ridges, binary, 15, 255, cv::THRESH_BINARY_INV);
	
	// Now upscale the clean, isolated line map for NFIQ 2
	// INTER_NEAREST preserves the clean, pixelated structural blocks without adding blur
	cv::Mat nfiqReady;
	cv::resize(binary, nfiqReady, cv::Size(500, 500), 0, 0, cv::INTER_NEAREST);
	
	std::string outName = "LOWRES_FIRST_" + rois.name;
	cv::imwrite(rois.name, nfiqReady);
}

void classicalEnhanceMassiveKernel(const Candidates& rois) {
	cv::Mat bgrImg = rois.fingerCrop;
	cv::Mat upscaled;
	cv::resize(bgrImg, upscaled, cv::Size(512, 512), 0, 0, cv::INTER_LANCZOS4);
	
	cv::Mat gray;
	cv::cvtColor(upscaled, gray, cv::COLOR_BGR2GRAY);
	
	//  Scale the kernel size way up (from 9x9 to 33x33) to bridge the upscaled gaps
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(33, 33));
	cv::Mat ridges;
	cv::morphologyEx(gray, ridges, cv::MORPH_BLACKHAT, kernel);
	
	// 2. Scale the CLAHE grid size up (from 8x8 to 32x32) to match the new macro-structures
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(32, 32));
	clahe->apply(ridges, ridges);
	
	// 3. Threshold and Invert
	cv::Mat binary;
	cv::threshold(ridges, binary, 25, 255, cv::THRESH_BINARY);
	
	cv::Mat finalOutput;
	cv::bitwise_not(binary, finalOutput);
	
	// 4. Final resize to 500x500 for NFIQ 2
	cv::Mat nfiqReady;
	cv::resize(finalOutput, nfiqReady, cv::Size(500, 500), 0, 0, cv::INTER_CUBIC);
	
//	std::string outName = "MASSIVE_KERNEL_" + rois.name;
	cv::imwrite(rois.name, nfiqReady);
}

void classicalEnhanceAndAnalyze(const Candidates& rois) {
	cv::Mat bgrImg = rois.fingerCrop;
	
	// Classical Upscaling (Bypass AI entirely)
	// Scale 64x64 directly to 512x512 using Lanczos4 interpolation (best for preserving edges)
	cv::Mat upscaled;
	cv::resize(bgrImg, upscaled, cv::Size(512, 512), 0, 0, cv::INTER_LANCZOS4);
	
	// Grayscale Conversion
	cv::Mat gray;
	cv::cvtColor(upscaled, gray, cv::COLOR_BGR2GRAY);
	
	// Extract Ridges using our Top-Hat/Black-Hat logic
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(9, 9));
	cv::Mat ridges;
	cv::morphologyEx(gray, ridges, cv::MORPH_BLACKHAT, kernel);
	
	// Boost Contrast
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
	clahe->apply(ridges, ridges);
	
	// Invert for NFIQ 2 (Black ridges on white background)
	cv::Mat finalOutput;
	cv::bitwise_not(ridges, finalOutput);
	
//	std::string outName = "CLASSICAL_" + rois.name;
//	std::cout<<"FinalOutput:"<<outName<<"\n";
	cv::imwrite(rois.name, finalOutput);
}

void testGaborFilterBank(const Candidates& rois) {
	cv::Mat gray;
	if (rois.fingerCrop.channels() >= 3) cv::cvtColor(rois.fingerCrop, gray, cv::COLOR_BGR2GRAY);
	else gray = rois.fingerCrop.clone();
	
	
	//  Moderate upscale to give the Gabor wavelets room to compute gradients
	cv::Mat upscaled;
	cv::resize(gray, upscaled, cv::Size(256, 256), 0, 0, cv::INTER_CUBIC);
	
	// Normalize localized contrast beforehand
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(4.0, cv::Size(8, 8));
	clahe->apply(upscaled, upscaled);
	
	// Create an accumulator canvas initialized to pure zero
	cv::Mat accumulatedResponses = cv::Mat::zeros(upscaled.size(), CV_32FC1);
	
	// Gabor Wavelet Hyperparameters (Tuned for typical ridge frequencies)
	int ksize = 15;        // Size of the filter bounding box
	double sigma = 2.5;    // Standard deviation of the Gaussian envelope
	double lambd = 6.0;    // Wavelength of the sinusoidal wave (ridge spacing)
	double gamma = 0.5;    // Spatial aspect ratio (elliptical shape of filter)
	double psi = 0.0;      // Phase offset
	
	// Scan 8 different angles (0 to 180 degrees) to catch ridges sweeping in any direction
	for (int i = 0; i < 8; i++) {
		double theta = i * CV_PI / 8.0; // Current angle iteration
		
		cv::Mat gaborKernel = cv::getGaborKernel(
												 cv::Size(ksize, ksize), sigma, theta, lambd, gamma, psi, CV_32F
												 );
		
		cv::Mat response;
		cv::filter2D(upscaled, response, CV_32F, gaborKernel);
		
		// Element-wise maximum: retain the strongest structural frequency response
		cv::max(accumulatedResponses, response, accumulatedResponses);
	}
	
	// Convert back to standard 8-bit Grayscale
	cv::Mat final8Bit;
	cv::normalize(accumulatedResponses, final8Bit, 0, 255, cv::NORM_MINMAX, CV_8UC1);
	
	// Invert so ridges display as dark lines on a white background (Standard for NFIQ 2)
	cv::bitwise_not(final8Bit, final8Bit);
	
	// Mandatory NFIQ 2 dimension compliance scale
	cv::Mat nfiqReady;
	cv::resize(final8Bit, nfiqReady, cv::Size(500, 500), 0, 0, cv::INTER_CUBIC);
	
//	std::string outName = "GABOR_" + rois.name;
	cv::imwrite(rois.name, nfiqReady);
}

void enhanceAndAnalyze(const Candidates& rois, torch::jit::script::Module& model, const torch::Device& device) {
	model.eval();
	
	cv::Mat bgrImg = rois.fingerCrop.clone();
	
	// --- TRACK A: PRE-AI CONTRAST INJECTION ---
	cv::Mat labImg;
	cv::cvtColor(bgrImg, labImg, cv::COLOR_BGR2Lab);
	
	std::vector<cv::Mat> labPlanes;
	cv::split(labImg, labPlanes);
	
	// Apply CLAHE specifically to the Luminance (brightness) channel
	// We use a small grid (4x4) because the input crop is only 64x64
	cv::Ptr<cv::CLAHE> preClahe = cv::createCLAHE(3.0, cv::Size(4, 4));
	preClahe->apply(labPlanes[0], labPlanes[0]);
	
	cv::merge(labPlanes, labImg);
	cv::cvtColor(labImg, bgrImg, cv::COLOR_Lab2BGR);
	
	// PURE PYTORCH INPUT CONVERSION (Bypasses opencv_lite convertTo entirely)
	// Read the 8-bit image directly into PyTorch
	torch::Tensor inputTensor = torch::from_blob(bgrImg.data, {1, 64, 64, 3}, torch::kByte);
	// Convert to float and divide by 255.0 inside the tensor natively
	inputTensor = inputTensor.to(torch::kFloat32).div(255.0);
	inputTensor = inputTensor.permute({0, 3, 1, 2}).contiguous().to(device);
	
	// RUN INFERENCE
	torch::Tensor outputTensor;
	try {
		torch::NoGradGuard no_grad;
		outputTensor = model.forward({inputTensor}).toTensor();
	}
	catch (const c10::Error& e) {
		std::cerr << "Model inference failed: " << e.what() << std::endl;
		return;
	}
	
	// OUTPUT EXTRACTION
	outputTensor = outputTensor.squeeze(0).permute({1, 2, 0}).cpu().contiguous();
	
	int height = outputTensor.size(0);
	int width = outputTensor.size(1);
	const float* tensorData = outputTensor.data_ptr<float>();
	
	// Allocate a 1-Channel GRAYSCALE canvas (CV_8UC1)
	cv::Mat hrImageGray(height, width, CV_8UC1);
	
	for (int i = 0; i < height; ++i) {
		uchar* rowPtr = hrImageGray.ptr<uchar>(i);
		for (int j = 0; j < width; ++j) {
			int tensorIdx = (i * width + j) * 3;
			
			// Extract the RGB float values from PyTorch and scale to 255
			float r = tensorData[tensorIdx + 0] * 255.0f;
			float g = tensorData[tensorIdx + 1] * 255.0f;
			float b = tensorData[tensorIdx + 2] * 255.0f;
			
			// Manual Grayscale conversion using the standard luminosity formula
			// This completely bypasses opencv_lite's broken color conversion routines
			float gray = 0.299f * r + 0.587f * g + 0.114f * b;
			
			// Write as a single grayscale byte
			rowPtr[j] = cv::saturate_cast<uchar>(gray);
		}
	}
	
	// --- DEBUG STEP: SAVE RAW AI OUTPUT ---
	std::string rawFileName = rois.name;
	size_t extPos = rawFileName.find(".png");
	if (extPos != std::string::npos) rawFileName.replace(extPos, 4, "_RAW.png");
	else rawFileName += "_RAW.png";
	
	// Because hrImageGray is 1-channel, imwrite will NOT attempt a color swap and will NOT crash.
	cv::imwrite(rawFileName, hrImageGray);
	// --------------------------------------
	
	// Now apply the enhancement
	cv::Mat enhancedFingertip = applyRidgeEnhancement2(hrImageGray);
	
	cv::Mat finalOutput;
	if (enhancedFingertip.type() != CV_8UC1)  enhancedFingertip.convertTo(finalOutput, CV_8UC1);
	else  finalOutput = enhancedFingertip;
	
	cv::Mat nfiqReady;
	cv::resize(finalOutput, nfiqReady, cv::Size(500, 500), 0, 0, cv::INTER_CUBIC);
	/*
	// SANITIZE AND SAVE ENHANCEMENT OUTPUT
	// Bypasses opencv_lite's convertTo tables for the final filtered matrix
	int outChannels = enhancedFingertip.channels();
	cv::Mat safeOutput8Bit(enhancedFingertip.rows, enhancedFingertip.cols, CV_8UC(outChannels));
	
	if (enhancedFingertip.type() == CV_32FC1 || enhancedFingertip.type() == CV_32FC3) {
		for (int i = 0; i < enhancedFingertip.rows; ++i) {
			const float* srcPtr = enhancedFingertip.ptr<float>(i);
			uchar* dstPtr = safeOutput8Bit.ptr<uchar>(i);
			for (int j = 0; j < enhancedFingertip.cols * outChannels; ++j) {
				// NOTE: If your final enhanced image is completely pitch black,
				// change "srcPtr[j]" below to "srcPtr[j] * 255.0f"
				dstPtr[j] = cv::saturate_cast<uchar>(srcPtr[j]);
			}
		}
	}
	else safeOutput8Bit = enhancedFingertip;
	*/
	std::cout << "writing enhanced Image to:" << rois.name << "\n";
	cv::imwrite(rois.name, finalOutput);
	
	//nfiq2 analysis should be placed here ...
}

void saveRawOutput(const Candidates& rois) {
	cv::Mat bgrImg = rois.fingerCrop.clone();
	cv::Mat upscaled;
	cv::Mat gray;
	cv::resize(bgrImg, upscaled, cv::Size(500, 500), 0, 0, cv::INTER_CUBIC);
	cv::cvtColor(upscaled, gray, cv::COLOR_BGR2GRAY);
	cv::imwrite(rois.name, gray);
}

void fingertipExtractor(const cv::Mat& frame,
						const std::vector<BoxKp3>& boxes,
						std::array<std::vector<Candidates>, 5>& fingerTips,
						torch::jit::script::Module& model,
						const torch::Device& device,
						const size_t frameNumber,
						const float padding = 0.2F)  {
	std::string fileName;
	fileName += std::to_string(frameNumber);
	cv::Point3f tip;
	cv::Point3f dip;
	size_t fingerCounter {1};
	size_t handCounter {0};
	
	for(const auto& box : boxes) {
		std::cout<<"Frame:"<<frameNumber<<", boxScore:"<<box.score<<", score prop:"<<(box.score < 0.99)<<"\n";
		if(box.score < 0.99) continue;
		
		if(const auto thumbRoi = getAlignedThumbRoi(frame, box); thumbRoi.has_value()) {
			fileName = outputDebugFingertipPath;
			fileName +=  std::to_string(frameNumber) + "_hand" + std::to_string(handCounter) + "_0.png";
			//enhanceAndAnalyze(Candidates{fileName, 0, thumbRoi.value()}, model, device);
//			testGaborFilterBank(Candidates{fileName, 0, thumbRoi.value()});
//			saveRawOutput(Candidates{fileName, 0, thumbRoi.value()});
			cv::imwrite(fileName, thumbRoi.value());
		}
		
		cv::Point3f wrist = box.points[0];
		cv::Point3f middleMcp = box.points[9];
		
		const float wristX = wrist.x * frame.size().width;
		const float wristY = wrist.y * frame.size().height;
		const float mcpX = middleMcp.x * frame.size().width;
		const float mcpY = middleMcp.y * frame.size().height;
		
		const float palmLength = std::hypotf(wristX - mcpX, wristY - mcpY);
		
		for(const auto& points : fingerPoints2) {
			cv::Mat roi;
			tip = box.points[points.first];
			dip = box.points[points.second];
			
			
			const double dynamicThreshold = palmLength * 0.15;
			if(const double score = getFingerScore(tip, dip, dynamicThreshold, frame.size().height); score <= 0.0) {
				std::cout<<"Frame:"<<frameNumber<<", finger:"<<fingerCounter<<", score:"<<score<<"\n";
				fingerCounter++;
				continue;
			}
			
			std::cout<<"Frame:"<<frameNumber<<", finger:"<<fingerCounter<<" score should be correct?\n";
			tip.x *= (float)frame.size().width;
			tip.y *= (float)frame.size().height;
			dip.x *= (float)frame.size().width;
			dip.y *= (float)frame.size().height;
			
			float fingerLength = std::hypotf(tip.x - dip.x, tip.y - dip.y);
			float boxSize = fingerLength*(1+padding*2);
			float x1 = std::max(0.F, tip.x - std::floor(boxSize/2.F));
			float y1 = std::max(0.F, tip.y - std::floor(boxSize/2.F));
			float x2 = std::min((float)frame.size().width, x1+boxSize);
			float y2 = std::min((float)frame.size().height, y1 + boxSize);
			float subMatrixWidth = x2 - x1;
			float subMatrixHeight = y2 - y1;
			frame(cv::Rect((int)x1, (int)y1, (int)subMatrixWidth, (int)subMatrixHeight)).copyTo(roi);
			
			// FORCE STRICT DIMENSIONS: 64x64
			cv::Mat standardizedRoi;
			cv::resize(roi, standardizedRoi, cv::Size(64, 64), 0, 0, cv::INTER_LINEAR);
			/*
			if (standardizedRoi.channels() == 1)
				cv::cvtColor(standardizedRoi, standardizedRoi, cv::COLOR_GRAY2RGB);
			else
				cv::cvtColor(standardizedRoi, standardizedRoi, cv::COLOR_BGR2RGB);
			*/
			fileName = outputDebugFingertipPath;
			fileName +=  std::to_string(frameNumber) + "_hand" + std::to_string(handCounter) + "_" + std::to_string(fingerCounter) + ".png";			
			
//			enhanceAndAnalyze(Candidates{fileName, fingerCounter, standardizedRoi}, model, device);
//			testGaborFilterBank(Candidates{fileName, fingerCounter, standardizedRoi});
			cv::imwrite(fileName, standardizedRoi);
//			saveRawOutput(Candidates{fileName, fingerCounter, standardizedRoi});
			fingerCounter++;
		}
		fingerCounter = 1;
		handCounter++;
	}
}

void printResults(const std::vector<std::vector<BoxKp3>>& results) {
	size_t frameNumber = 0;
	for(const auto& resultFrame: results) {
		for(const auto& box: resultFrame) {
			std::cout<<"Frame:"<<frameNumber<<" Rect ["<<box.rect.x<<", "<<box.rect.y<<"], "<<box.rect.width<<", "<<box.rect.height<<", radians:"<<box.radians<<", score:"<<box.score<<", points:";
			for(const auto& point : box.points)
				std::cout<<"["<<point.x<<", "<<point.y<<", "<<point.z<<"] ";
			std::cout<<"\n";
		}
		std::cout<<"\n\n";
		frameNumber++;
	}
}

//cv::imwrite
void runTest(const std::string& filePath, const std::string& modelPath) noexcept {
	std::vector<std::vector<BoxKp2>> results = readFingerFromVideoFile(filePath, modelPath);
	size_t frameNumber = 0;
	for(const auto& resultFrame: results) {
		for(const auto& box: resultFrame) {
			std::cout<<"Frame:"<<frameNumber<<" Rect ["<<box.rect.x<<", "<<box.rect.y<<"], "<<box.rect.width<<", "<<box.rect.height<<", radians:"<<box.radians<<", score:"<<box.score<<", classID:"<<box.classId<<", points:";
			for(const auto& point : box.points)
				std::cout<<"["<<point.x<<", "<<point.y<<"] ";
			std::cout<<"\n";
		}
		std::cout<<"\n\n";
		frameNumber++;
	}
}

void runLandmarker(const std::string& filePath, const std::string& palmDetectionModelPath, const std::string& landmarkerModelPath) {
	std::vector<std::vector<BoxKp3>> results = getHandsLandmarks(filePath, palmDetectionModelPath, landmarkerModelPath);
	printResults(results);
}

const std::vector<std::vector<BoxKp2>> readFingerFromVideoFile(const std::string& filePath, const std::string& modelPath) noexcept {
	std::vector<std::vector<BoxKp2>> results;
	cv::VideoCapture capture;
	cv::Mat frame;
	mpp::HandDetector detector(modelPath);
	
	capture.open(filePath);
	size_t frameNumber = 0;
	while(true) {
		if(!capture.read(frame)) break;
		std::vector<BoxKp2> output = {};
		detector.run(frame, output);
		results.emplace_back(output);
		drawHands(frame, output, frameNumber);
		frameNumber++;
	}
	return results;
}

[[nodiscard]] const std::vector<BoxKp3> getHandLandmarksfromSingleImage(const std::string singeImagePath, const std::string& palmDetectionModelPath, const std::string& landmarkerModelPath, mpp::HandLandmarker& landmarker, std::array<std::vector<Candidates>, 5>& rois, torch::jit::script::Module& model, const torch::Device& device) {
	cv::Mat frame = cv::imread(singeImagePath, cv::IMREAD_COLOR);
	std::vector<BoxKp3> output = {};
	landmarker.runImage(frame, output);
	fingertipExtractor(frame, output, rois, model, device, 0);
	return output;
}

void rundLandmarkerOnImageDirectory(const std::filesystem::path& directoryPath, const std::string& palmDetectionModelPath, const std::string& ladmarkerModelPath) {
	std::vector<std::vector<BoxKp3>> results;
	std::array<std::vector<Candidates>, 5> rois;
	for(auto& roi : rois)
		roi.reserve(15);
	torch::jit::script::Module model = torch::jit::load(modelFreezenPath);
	torch::Device device(torch::kCPU);
	model.to(device);
	mpp::HandLandmarker landmarker(palmDetectionModelPath, ladmarkerModelPath, 4);
	for(const auto& dirEntry : std::filesystem::directory_iterator{directoryPath}) {
		results.emplace_back(getHandLandmarksfromSingleImage(dirEntry.path().string(), palmDetectionModelPath, ladmarkerModelPath, landmarker, rois, model, device));
	}
	printResults(results);
}

void runLandmarkerOnSingeImagefile(const std::string& filePath, const std::string& palmDetectionModelPath, const std::string& ladmarkerModelPath) {
	std::vector<std::vector<BoxKp3>> results;
	std::array<std::vector<Candidates>, 5> rois;
	for(auto& roi : rois)
		roi.reserve(15);
	torch::jit::script::Module model = torch::jit::load(modelFreezenPath);
	torch::Device device(torch::kCPU);
	model.to(device);
	mpp::HandLandmarker landmarker(palmDetectionModelPath, ladmarkerModelPath, 4);
	
	results.emplace_back(getHandLandmarksfromSingleImage(filePath, palmDetectionModelPath, ladmarkerModelPath, landmarker, rois, model, device));
	printResults(results);
}

const std::vector<std::vector<BoxKp3>> getHandsLandmarks(const std::string& filePath, const std::string& palmDetectionModelPath, const std::string& ladmarkerModelPath) {
	std::cout <<"TORCH; fucking metal available:" <<torch::mps::is_available() << std::endl;
	std::vector<std::vector<BoxKp3>> results;
//	std::vector<Candidates> rois;
	std::array<std::vector<Candidates>, 5> rois;
	for(auto& roi : rois)
		roi.reserve(15);
	torch::jit::script::Module model = torch::jit::load(modelFreezenPath);
	torch::Device device(torch::kCPU);
	model.to(device);
//	model.eval(); //MacOS
//	rois.reserve(15);
	cv::VideoCapture capture(filePath, cv::CAP_AVFOUNDATION);
	cv::Mat frame;
	mpp::HandLandmarker landmarker(palmDetectionModelPath, ladmarkerModelPath, 4);
//	capture.open(filePath);
	size_t frameNumber = 0;
	while (true) {
		if(!capture.read(frame)) break;
		std::vector<BoxKp3> output = {};
		landmarker.runImage(frame, output);
		results.emplace_back(output);
		fingertipExtractor(frame, output, rois, model, device, frameNumber);
//		drawHandLandmark(frame, output, frameNumber);
		frameNumber++;
	}
	return results;
}
