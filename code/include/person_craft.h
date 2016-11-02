#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

namespace person_craft {
	void* scene_feature_load_model(std::string, std::string, std::string = "pool5/7x7_s1");
	std::vector<float> extract_scene_features(const cv::Mat img, void* ptr);
}


