#include <vector>
#include <vector>
#include "caffe/layers/operator/parallel_batch_norm_layer.hpp"
#include "caffe/util/math_functions.hpp"
#define BN_EPS 1e-5

namespace caffe {


static __global__ void kernel_test_forward(
    const int num, const int channels, const int spatial_dim,
    const float* scale, const float* bias, const float* mean, const float* var, 
    const float eps, const float* bottom_data,  float* top_data) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int c = (index / spatial_dim) % channels;
    top_data[index] = ((bottom_data[index] - mean[c]) / sqrt(var[c] + eps))
        * scale[c] + bias[c];
  }
}


static __global__ void kernel_test_backward(
    const int num, const int channels, const int spatial_dim,
    const float* scale, const float* bias, const float* mean, const float* var,
    const float eps, const float* top_diff, float* bottom_diff) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int c = (index / spatial_dim) % channels;
    bottom_diff[index] = top_diff[index] / sqrt(var[c] + eps) * scale[c];
  }
}


static __global__ void kernel_local_stats(int num, int channels, int spatial_dim,
    const float norm_factor,
    const float* bottom_data, float* mean, float* var) {
  // store local E[x] to mean, E[x^2] to var temporarily
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) {
    const int index = i / spatial_dim * channels * spatial_dim
        + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += bottom_data[index];
    buffer2[tid] += bottom_data[index] * bottom_data[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) {
    mean[c] = buffer1[0] / norm_factor;
    var[c] = buffer2[0] / norm_factor;
  }
}


static __global__ void kernel_backward_scale_bias(
    const int num, const int channels, const int spatial_dim,
    const float* mean, const float* var, const float eps,
    const float* top_diff, const float* bottom_data,
    float* scale_diff, float* bias_diff) {
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) {
    const int index = i / spatial_dim * channels * spatial_dim
        + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += top_diff[index] * (bottom_data[index] - mean[c]) / sqrt(var[c] + eps);
    buffer2[tid] += top_diff[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      buffer1[tid] += buffer1[tid + s];
      buffer2[tid] += buffer2[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) {
    scale_diff[c] = buffer1[0];
    bias_diff[c] = buffer2[0];
  }
}


static __global__ void kernel_backward_bottom(
    const int num, const int channels, const int spatial_dim,
    const float* scale, const float* bias,
    const float* mean, const float* var, const float eps,
    const float norm_factor,
    const float* top_diff, const float* scale_diff, const float* bias_diff,
    const float* bottom_data, float* bottom_diff) {
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) {
    int c = (index / spatial_dim) % channels;
    const float inv_std = float(1) / sqrt(var[c] + eps);
    const float x_norm = (bottom_data[index] - mean[c]) * inv_std;
    bottom_diff[index] = scale[c] * inv_std *
        (top_diff[index] - (x_norm * scale_diff[c] + bias_diff[c]) / norm_factor);
  }
}


void ParallelBatchNormLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (Caffe::number_collect_sample != -1)
	{
		CHECK_EQ(this->parallel_blobs_.size(),4*NGPUS);
		if (Caffe::number_collect_sample == 0)
		{
			caffe_gpu_set(this->blobs_[2]->count(),float(0),this->blobs_[2]->mutable_gpu_data());
			caffe_gpu_set(this->blobs_[3]->count(),float(0),this->blobs_[3]->mutable_gpu_data());
		}		
		for (int i = 0; i < NGPUS; i++) 
		{  	
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
			ncclBcast((void *)this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[2*NGPUS+i]->count(),
	 																		ncclFloat,0,Caffe::comms(i),NULL);			
		}		
		for (int i = 0; i < NGPUS; i++) 
		{  	
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
			ncclBcast((void *)this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[3*NGPUS+i]->count(),
	 																		ncclFloat,0,Caffe::comms(i),NULL);			
		}		
	}	 
#if 0
	for (int i = 0; i < bottom.size(); ++i) 
  {  	
  	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
  	ncclBcast((void *)this->parallel_blobs_[0*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[0*NGPUS+i]->count(),
 																		ncclFloat,0,Caffe::comms(i),Caffe::parallel_stream(i));											
	}
	for (int i = 0; i < bottom.size(); ++i) 
  {  	
  	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));																		
 		ncclBcast((void *)this->parallel_blobs_[1*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[1*NGPUS+i]->count(),
 																		ncclFloat,0,Caffe::comms(i),Caffe::parallel_stream(i));																																
	}
	for (int i = 0; i < bottom.size(); ++i) 
  {  	
  	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));																			
 		ncclBcast((void *)this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[2*NGPUS+i]->count(),
 																		ncclFloat,0,Caffe::comms(i),Caffe::parallel_stream(i));																																
	}
	for (int i = 0; i < bottom.size(); ++i) 
  {  	
  	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));																		
 		ncclBcast((void *)this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[3*NGPUS+i]->count(),
 																		ncclFloat,0,Caffe::comms(i),Caffe::parallel_stream(i));																																
	}
#endif

	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	const int m = num * height * width * NGPUS;
//----------------------------------------------------	
	// compute local E[x] and E[x^2]
	if (Caffe::bn_state() == "learned")
	{	
		if (Caffe::number_collect_sample == 0)
		{
			for(int i=0;i<NGPUS;i++)
			{ 
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
				caffe_gpu_set(this->parallel_blobs_[2*NGPUS+i]->count(),float(0),this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data());
				caffe_gpu_set(this->parallel_blobs_[3*NGPUS+i]->count(),float(0),this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data());
			}
		}
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
			kernel_local_stats<<<channels, CAFFE_CUDA_NUM_THREADS>>>(
				num, channels, height * width,
				float(m),
				bottom[i]->gpu_data(),
				parallel_mean_buffer_[i]->mutable_gpu_data(),
				parallel_var_buffer_[i]->mutable_gpu_data());
		}		
		// sync E[x] and E[x^2]
		REDUCE_DATA(parallel_mean_buffer_);
		REDUCE_DATA(parallel_var_buffer_);


		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
			caffe_gpu_mul(channels, parallel_mean_buffer_[i]->gpu_data(), parallel_mean_buffer_[i]->gpu_data(),
							      top[i]->mutable_gpu_data());  // reuse the top buffer
			caffe_gpu_sub(channels, parallel_var_buffer_[i]->gpu_data(), top[i]->gpu_data(),
							      parallel_var_buffer_[i]->mutable_gpu_data());
		}			 
		float factor;
		if (Caffe::number_collect_sample == -1)
			factor = 0.01;
		else 
			factor = float(1)/float(Caffe::number_collect_sample+1);
		
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
			caffe_gpu_axpby(parallel_mean_buffer_[i]->count(),
	      factor, parallel_mean_buffer_[i]->gpu_data(),
	      1-factor, this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data());
			caffe_gpu_axpby(parallel_var_buffer_[i]->count(),
		    factor, parallel_var_buffer_[i]->gpu_data(),
		    1-factor, this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data());
		}
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
			kernel_test_forward<<<CAFFE_GET_BLOCKS(bottom[i]->count()),CAFFE_CUDA_NUM_THREADS>>>
			( num, channels, height * width,
				this->parallel_blobs_[0*NGPUS+i]->gpu_data(),
				this->parallel_blobs_[1*NGPUS+i]->gpu_data(),
				parallel_mean_buffer_[i]->gpu_data(),
				parallel_var_buffer_[i]->gpu_data(),
				float(BN_EPS),
				bottom[i]->gpu_data(),
				top[i]->mutable_gpu_data());
		}
	}
	else
	{
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
			kernel_test_forward<<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
			( num, channels, height * width,
				this->parallel_blobs_[0*NGPUS+i]->gpu_data(),
				this->parallel_blobs_[1*NGPUS+i]->gpu_data(),
				this->parallel_blobs_[2*NGPUS+i]->gpu_data(),
				this->parallel_blobs_[3*NGPUS+i]->gpu_data(),
				float(BN_EPS),
				bottom[i]->gpu_data(),
				top[i]->mutable_gpu_data());
		}
	}
//----------------------------------------------------
		
	
	if (Caffe::number_collect_sample != -1)
	{
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
			ncclReduce( this->parallel_blobs_[2*NGPUS+i]->gpu_data(),this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),
					this->parallel_blobs_[2*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
		}
		for(int i=0;i<NGPUS;i++)
		{ 
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
			ncclReduce( this->parallel_blobs_[3*NGPUS+i]->gpu_data(),this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),
					this->parallel_blobs_[3*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
		}
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
		caffe_gpu_scal(this->blobs_[2]->count(),float(1)/float(NGPUS),this->blobs_[2]->mutable_gpu_data());
		caffe_gpu_scal(this->blobs_[3]->count(),float(1)/float(NGPUS),this->blobs_[3]->mutable_gpu_data());	
	}			
}


void ParallelBatchNormLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{	 
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width(); 
	
	// compute local scale and bias diff
	for(int i=0;i<NGPUS;i++)
	{ 
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
		kernel_backward_scale_bias<<<channels, CAFFE_CUDA_NUM_THREADS>>>(
			num, channels, height * width,
			parallel_mean_buffer_[i]->gpu_data(),
			parallel_var_buffer_[i]->gpu_data(),
			float(BN_EPS),
			top[i]->gpu_diff(),
			bottom[i]->gpu_data(),
			parallel_mean_buffer_[i]->mutable_gpu_diff(),  // temp use for local scale diff
			parallel_var_buffer_[i]->mutable_gpu_diff() // temp use for local bias diff
		);
	}
	// sync scale and bias diff
	REDUCE_DIFF(parallel_mean_buffer_)
	REDUCE_DIFF(parallel_var_buffer_);
	// add to param blobs diff
	for(int i=0;i<NGPUS;i++)
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
		caffe_gpu_axpy(channels, float(1) / float(NGPUS),
			             parallel_mean_buffer_[i]->gpu_diff(),
			             this->parallel_blobs_[0*NGPUS+i]->mutable_gpu_diff());
		caffe_gpu_axpy(channels, float(1) / float(NGPUS),
			             parallel_var_buffer_[i]->gpu_diff(),
			             this->parallel_blobs_[1*NGPUS+i]->mutable_gpu_diff());         	             
	}	               
	// compute bottom diff
	for(int i=0;i<NGPUS;i++)
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));	
		kernel_backward_bottom<<<CAFFE_GET_BLOCKS(bottom[i]->count()),
			  CAFFE_CUDA_NUM_THREADS>>>(
			num, channels, height * width,
			this->parallel_blobs_[0*NGPUS+i]->gpu_data(),
			this->parallel_blobs_[1*NGPUS+i]->gpu_data(),
			parallel_mean_buffer_[i]->gpu_data(),
			parallel_var_buffer_[i]->gpu_data(),
			float(BN_EPS),
			float(num * height * width * NGPUS),
			top[i]->gpu_diff(),
			parallel_mean_buffer_[i]->gpu_diff(),
			parallel_var_buffer_[i]->gpu_diff(),
			bottom[i]->gpu_data(),
			bottom[i]->mutable_gpu_diff());
	}
}

void ParallelBatchNormLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
