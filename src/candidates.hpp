//
//  candidates.hpp
//  Fingerprint
//
//  Created by Luis Valdivia on 15.05.26.
//

#pragma once
#include "base_type.h"
#include <opencv2/opencv.hpp>
#include <array>
#include <string_view>

struct Candidates {
	const std::string name;
	const size_t fingerNumber; // 0 thumb, 1 index, 2 middle, 3 anular, 4 small
	cv::Mat fingerCrop;
};


/* candidates_h */
