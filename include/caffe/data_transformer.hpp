#ifndef CAFFE_DATA_TRANSFORMER_HPP
#define CAFFE_DATA_TRANSFORMER_HPP

#include <vector>


#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Applies common transformations to the input data, such as
 * scaling, mirroring, substracting the image mean...
 */

class DataTransformer {
public:
	explicit DataTransformer(const TransformationParameter& param);
	virtual ~DataTransformer() {}

	virtual int Rand(int n);



	void Transform(const cv::Mat& cv_img, Blob* transformed_blob);
	void Transformsimple(const cv::Mat& cv_img, Blob* transformed_blob);
	void TransformImgAndSeg(const std::vector<cv::Mat>& cv_img_seg,
													Blob* transformed_data_blob, Blob* transformed_label_blob,
													const int ignore_label);
	void TransformGan(const std::vector<cv::Mat>& cv_img_seg,
													Blob* transformed_data_blob, Blob* transformed_label_blob,
													const int ignore_label);												

	void TransformParse(const cv::Mat& cv_img, std::vector<float> label, Blob* transformed_blob,Blob* transformed_label);


protected:



	// Tranformation parameters
	TransformationParameter param_;


	Blob data_mean_;
	vector<float> mean_values_;
};

}  // namespace caffe

#endif  // CAFFE_DATA_TRANSFORMER_HPP_
