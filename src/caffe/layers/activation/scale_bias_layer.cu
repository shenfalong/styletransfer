
#include <vector>

#include "caffe/layers/activation/scale_bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void forward_kernel(const int count, const int channels, const int spatial_dim, const int n_labels, 
																												const float* in, const float * label, 
																												const float * w, const float * b, float* out) 
{
  CUDA_KERNEL_LOOP(i, count) 
  {
  	int n = i / spatial_dim / channels;
  	int c = i / spatial_dim % channels;
  	int cur_label = label[n];
  	
  	out[i] = w[cur_label*channels+c] * in[i] + b[cur_label*channels+c];
  }
}

static __global__ void backward_kernel_data(const int count, const int channels, const int spatial_dim, const int n_labels, 
																												const float* out_diff, const float *label,
																												const float * w, const float * b, float* in_diff) 
{
  CUDA_KERNEL_LOOP(i, count) 
  {
  	int n = i / spatial_dim / channels;
  	int c = i / spatial_dim % channels;
  	int cur_label = label[n];
  	
  	in_diff[i] = w[cur_label*channels+c] * out_diff[i];
  }
}

static __global__ void backward_kernel_weight(int channels, int spatial_dim, const int n_labels, 
																											 const float* top_diff, const float * bottom_data,  const int cur_label,
																											 float* w_diff, float* b_diff) 
{
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < spatial_dim; i += blockDim.x) 
  {
    const int index =   c * spatial_dim + i;
    buffer1[tid] += top_diff[index] * bottom_data[index];
    buffer2[tid] += top_diff[index];
  }
  __syncthreads();
  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    w_diff[cur_label*channels+c] += buffer1[0];
    b_diff[cur_label*channels+c] += buffer2[0];
  }
}
//---------------------------

static __global__ void mean_abstract_kernel(int count, int channels, int spatial_dim, const float * in, float *out)
{
	CUDA_KERNEL_LOOP(i, count)
  {
  	int c = i / spatial_dim % channels;
  	if (c == 0)
  		out[i] = in[i] - 104.008;
  	else if (c == 1)
  		out[i] = in[i] - 116.669;
  	else
  		out[i] = in[i] - 122.675;
  }
}
//---------------------------

void ScaleBiasLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (true)
	{
		caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
		caffe_gpu_scal(top[0]->count(), float(127.5), top[0]->mutable_gpu_data());
		caffe_gpu_add_scalar(top[0]->count(), float(127.5), top[0]->mutable_gpu_data());
	
		mean_abstract_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[0]->count(),top[0]->channels(),top[0]->height()*top[0]->width(),top[0]->gpu_data(),top[0]->mutable_gpu_data());
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		forward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
  	(bottom[0]->count(), channels, height*width, classes_, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 
  						this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(), top[0]->mutable_gpu_data());
	}
}


void ScaleBiasLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (true)
	{
		caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
		caffe_gpu_scal(bottom[0]->count(), float(127.5), bottom[0]->mutable_gpu_diff());
	}
	else
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		backward_kernel_data<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), channels, height*width, classes_, top[0]->gpu_diff(), bottom[1]->gpu_data(), 
									this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(), bottom[0]->mutable_gpu_diff());
		
		if (this->lr_mult()[0] > 0 && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false)
		{		
			for (int n=0;n<num;n++)
			{
				backward_kernel_weight<<<channels,CAFFE_CUDA_NUM_THREADS>>>
				(channels, height*width, classes_, top[0]->gpu_diff()+top[0]->offset(n), bottom[0]->gpu_data()+bottom[0]->offset(n), int(bottom[1]->cpu_data()[n]),
				this->blobs_[0]->mutable_gpu_diff(), this->blobs_[1]->mutable_gpu_diff()); 
			}
		}
	}

}

void ScaleBiasLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (true)
	{
		caffe_copy(bottom[0]->count(),bottom[0]->gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
		caffe_gpu_scal(bottom[0]->count(), float(127.5), top[0]->mutable_gpu_sec_diff());
	}
	else
	{
	/*
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
				
		backward_kernel_data<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->count(), channels, height*width, bottom[0]->gpu_sec_diff(), this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(), top[0]->mutable_gpu_sec_diff());
	
		if (this->lr_mult()[0] > 0 && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false)
		{		
			backward_kernel_weight<<<channels,CAFFE_CUDA_NUM_THREADS>>>
			(channels, height*width,  bottom[0]->gpu_sec_diff(), top[0]->gpu_diff(), this->blobs_[0]->mutable_gpu_diff(), this->blobs_[1]->mutable_gpu_sec_diff()); 
		}
	*/	
	}
}

}  // namespace caffe
