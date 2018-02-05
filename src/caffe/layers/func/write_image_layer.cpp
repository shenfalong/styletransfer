
#include <vector>

#include "caffe/layers/func/write_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void WriteImageLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
}


void WriteImageLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}



REGISTER_LAYER_CLASS(WriteImage);
}  // namespace caffe
