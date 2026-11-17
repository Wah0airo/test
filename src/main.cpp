//
//  main.cpp
//  Fingerprint
//
//  Created by Luis Valdivia on 08.05.26.
//

#include <iostream>
#include "videoHandDetector.hpp"

int main([[maybe_unused]] int argc, [[maybe_unused]] const char * argv[]) {
//	runTest("/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/videos/one/part1.mp4", "/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/mediapipe_cmake/hand_detector/models/palm_detection_full.mnn");
//	runLandmarker("/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/videos/one/part1.mp4", "/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/mediapipe_cmake/hand_detector/models/palm_detection_full.mnn", "/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/mediapipe_cmake/hand_landmarker/models/hand_landmark_full.mnn");
	
	runLandmarkerOnSingeImagefile("/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/readplus1hand/selection/u002_hand_truck_vis_001.jpg", "/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/mediapipe_cmake/hand_detector/models/palm_detection_full.mnn", "/Users/luisvaldivia/Documents/uniSalzburg_1/MultimediaSeminar/mediapipe_cmake/hand_landmarker/models/hand_landmark_full.mnn");
	return EXIT_SUCCESS;
}
