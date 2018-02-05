
#include <vector>

#include "caffe/layers/operator/mean_variance_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define BN_EPS 1e-5
namespace caffe {


static __global__ void mean_variance_forward(int channels,int spatial_dim, float inv_norm_factor, const float* bottom_data, float* top_data) 
{
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x % channels;
  const int n = blockIdx.x / channels;


  buffer1[tid] = 0;
  buffer2[tid] = 0;
  for (int i = tid; i < spatial_dim; i += blockDim.x) 
  {
    const int index = blockIdx.x * spatial_dim + i;
    buffer1[tid] += bottom_data[index];
    buffer2[tid] += bottom_data[index] * bottom_data[index];
  }
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) 
  {
    top_data[n*2*channels+         c] = buffer1[0] * inv_norm_factor;
    top_data[n*2*channels+channels+c] = buffer2[0] * inv_norm_factor;
  }
}

static __global__ void square_root(int count, int channels,const float * mean_square, float * mean_var)
{
	CUDA_KERNEL_LOOP(i, count)
  {
  	const int c = i % channels;
  	const int n = i / channels;
  
  	mean_var[n*2*channels+c] = mean_square[n*2*channels+c];
  	
		mean_var[n*2*channels+channels+c] = sqrt(mean_square[n*2*channels+channels+c] - mean_square[n*2*channels+c]*mean_square[n*2*channels+c] + BN_EPS);
	}
}


static __global__ void mean_variance_backward(int count, int channels, int spatial_dim, float inv_norm_factor,
    const float* top_diff, const float* top_data, const float* bottom_data, float* bottom_diff) 
{
	CUDA_KERNEL_LOOP(i, count)
  {
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		
		bottom_diff[i] = top_diff[n*2*channels+c]*inv_norm_factor + 
										 top_diff[n*2*channels+channels+c]*inv_norm_factor*(bottom_data[i]-top_data[n*2*channels+c])/top_data[n*2*channels+channels+c];
	}
}

void MeanVarianceLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	mean_variance_forward<<<num*channels, CAFFE_CUDA_NUM_THREADS>>>
	(channels, height*width, float(1)/float(height*width), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
	
	square_root<<<CAFFE_GET_BLOCKS(num*channels), CAFFE_CUDA_NUM_THREADS>>>
	(num*channels,channels,top[0]->gpu_data(),top[0]->mutable_gpu_data());
}


void MeanVarianceLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
 
	mean_variance_backward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(), channels, height * width, float(1)/float(height*width),top[0]->gpu_diff(), top[0]->gpu_data(),bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
}

void MeanVarianceLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
