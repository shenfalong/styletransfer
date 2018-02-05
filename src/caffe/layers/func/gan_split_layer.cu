
#include <vector>

#include "caffe/layers/func/gan_split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {




void GANSplitLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
	
	if (Caffe::gan_type() == "train_dnet")
	{
		caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[1]->mutable_gpu_data());
	}
	else
	{
		caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[1]->mutable_gpu_data());
		caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[1]->mutable_gpu_data()+bottom[0]->count());
	}
}


void GANSplitLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::gan_type() == "train_dnet")
	{
		caffe_gpu_add(bottom[0]->count(), float(1), top[0]->gpu_diff(), 
																			float(1),top[1]->gpu_diff(),  
																			bottom[0]->mutable_gpu_diff());
	}
	else
	{
		caffe_gpu_add(bottom[0]->count(), float(1), top[0]->gpu_diff(), 
																			float(1),top[1]->gpu_diff(),  
																			bottom[0]->mutable_gpu_diff());
		caffe_gpu_add(bottom[0]->count(), float(1), bottom[0]->gpu_diff(), 
																			float(1),top[1]->gpu_diff()+bottom[0]->count(),  
																			bottom[0]->mutable_gpu_diff());
	}
}


void GANSplitLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
