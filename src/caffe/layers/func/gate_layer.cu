
#include <vector>

#include "caffe/layers/func/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


static __global__ void gate_forward(int count,const float * in_0, const float * in_1, float * out)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		out[i] = in_0[i] * in_1[i];		
	}
}


static __global__ void gate_backward(int count,const float * diff_out,const float *in_0, const float *in_1, float * diff_in_0, float * diff_in_1)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		diff_in_0[i] = diff_out[i] * in_1[i];
		diff_in_1[i] = diff_out[i] * in_0[i];
	}
}


void GateLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	gate_forward<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());	
}


void GateLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	gate_backward<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(), top[0]->gpu_diff(),bottom[0]->gpu_data(),bottom[1]->gpu_data(),
								bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());		
}


void GateLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
