//
//  videoHandDetector.hpp
//  Fingerprint
//
//  Created by Luis Valdivia on 12.05.26.
//

#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "candidates.hpp"

void runTest(const std::string& filePath, const std::string& modelPath) noexcept;

void runLandmarker(const std::string& filePath, const std::string& palmDetectionModelPath, const std::string& ladmarkerModelPath);

void rundLandmarkerOnImageDirectory(const std::filesystem::path& directoryPath, const std::string& palmDetectionModelPath, const std::string& ladmarkerModelPath);

void runLandmarkerOnSingeImagefile(const std::string& filePath, const std::string& palmDetectionModelPath, const std::string& ladmarkerModelPath);

[[nodiscard]] const std::vector<std::vector<BoxKp2>> readFingerFromVideoFile(const std::string& filePath, const std::string& modelPath) noexcept;

[[nodiscard]] const std::vector<std::vector<BoxKp3>> getHandsLandmarks(const std::string& filePath, const std::string& palmDetectionModelPath, const std::string& landmarkerModelPath);

/* videoHandDetector_hpp */


/*
 void fingertipExtractor(const cv::Mat& frame,
 const std::vector<BoxKp3>& boxes,
 std::array<std::vector<Candidates>, 5>& fingerTips,
 torch::jit::script::Module& model,
 const torch::Device& device,
 const size_t frameNumber,
 const float padding = 0.2F)
 
 */
