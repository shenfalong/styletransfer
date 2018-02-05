
#include <vector>

#include "caffe/layers/activation/bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void forward_kernel(const int count, const int channels, const int spatial_dim, const float* in, const float * b, float* out) 
{
  CUDA_KERNEL_LOOP(i, count) 
  {
  	int c = i / spatial_dim % channels;
  	out[i] = in[i] + b[c];
  }
}


static __global__ void backward_kernel_bias(int num, int channels, int spatial_dim,  const float* top_diff, float* b_diff) 
{
  __shared__ float buffer[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer[tid] += top_diff[index];
  }
  __syncthreads();
  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer[tid] += buffer[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    b_diff[c] += buffer[0];
  }
}

void BiasLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	forward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), channels, height*width, bottom[0]->gpu_data(), this->blobs_[0]->gpu_data(), top[0]->mutable_gpu_data());
	
}


void BiasLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
	
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{		
		backward_kernel_bias<<<channels,CAFFE_CUDA_NUM_THREADS>>>
		(num, channels, height*width,  top[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff()); 
	}
}

void BiasLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
				
	caffe_copy(bottom[0]->count(),bottom[0]->gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
}

}  // namespace caffe
