
#include <vector>

#include "caffe/layers/func/concat_num_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void ConcatNumLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	float * top_data = top[0]->mutable_gpu_data();
	caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top_data);
	top_data += bottom[0]->count();

	caffe_copy(bottom[1]->count(),bottom[1]->gpu_data(),top_data);
	top_data += bottom[1]->count();

	if (bottom.size() == 3)
		caffe_copy(bottom[2]->count(),bottom[2]->gpu_data(),top_data);
}


void ConcatNumLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	const float * top_diff = top[0]->mutable_gpu_diff();
	
	caffe_copy(bottom[0]->count(),top_diff,bottom[0]->mutable_gpu_diff());
	top_diff += bottom[0]->count();
	
	caffe_copy(bottom[1]->count(),top_diff,bottom[1]->mutable_gpu_diff());
	top_diff += bottom[1]->count();
	
	if (bottom.size() == 3)
		caffe_copy(bottom[2]->count(),top_diff,bottom[2]->mutable_gpu_diff());
}

void ConcatNumLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	float * top_sec_diff = top[0]->mutable_gpu_sec_diff();
	caffe_copy(bottom[0]->count(),bottom[0]->gpu_sec_diff(),top_sec_diff);
	top_sec_diff += bottom[0]->count();

	caffe_copy(bottom[1]->count(),bottom[1]->gpu_sec_diff(),top_sec_diff);
	top_sec_diff += bottom[1]->count();

	if (bottom.size() == 3)
		caffe_copy(bottom[2]->count(),bottom[2]->gpu_sec_diff(),top_sec_diff);
}

}  // namespace caffe
