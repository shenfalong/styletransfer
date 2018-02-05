#include <algorithm>
#include <vector>

#include "caffe/layers/activation/elu_layer.hpp"

namespace caffe {


static __global__ void forward_kernel(const int n, const float* in, float* out) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    out[index] = in[index] > 0 ? in[index] : exp(in[index]) - 1;
  }
}


static __global__ void backward_kernel_0(const int n, const float* out_diff, const float* in_data, float* in_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    in_diff[index] = in_data[index] > 0 ? out_diff[index] :out_diff[index] * exp(in_data[index]);
  }
}

static __global__ void backward_kernel_1(const int n, const float* out_diff, const float* in_data, float* in_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    in_diff[index] += in_data[index] > 0 ? out_diff[index] :out_diff[index] * exp(in_data[index]);
  }
}

static __global__ void secforward_kernel_diff(const int n, const float* in_sec_diff, const float* in_data, float* out_sec_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    out_sec_diff[index] = in_data[index] > 0 ? in_sec_diff[index] :in_sec_diff[index] * exp(in_data[index]);
  }
}

static __global__ void secforward_kernel_data(const int n, const float* in_sec_diff, const float * out_diff, const float* in_data, float* in_diff) {
  CUDA_KERNEL_LOOP(index, n) 
  {
    in_diff[index] = in_data[index] > 0 ? 0 :in_sec_diff[index] * out_diff[index] * exp(in_data[index]);
  }
}

void ELULayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	forward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
	CUDA_POST_KERNEL_CHECK;
}


void ELULayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (this->has_bottom_sec_diff_ ==  false)
	{
		backward_kernel_0<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
	}
	else
	{
		backward_kernel_1<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
		this->has_bottom_sec_diff_ = false;
	}	
}


void ELULayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

	secforward_kernel_diff<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), bottom[0]->gpu_sec_diff(),  bottom[0]->gpu_data(), top[0]->mutable_gpu_sec_diff());
	
	secforward_kernel_data<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), bottom[0]->gpu_sec_diff(), top[0]->gpu_diff(),  bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
	this->has_bottom_sec_diff_ = true;
}


}  // namespace caffe

