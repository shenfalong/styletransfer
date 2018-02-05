#include <algorithm>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/operator/batch_norm_layer.hpp"
//#include "caffe/layers/operator/cudnn_batch_norm_layer.hpp"
#define BN_EPS float(1e-5)

namespace caffe {
//---------------------------------- forward ---------------

static __global__ void kernel_local_stats(int num, int channels, int spatial_dim, const float norm_factor, const float* bottom_data, 
																					float* mean, float* var) 
{
  // store local E[x] to mean, E[x^2] to var temporarily
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += bottom_data[index];
    buffer2[tid] += bottom_data[index] * bottom_data[index];
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
    mean[c] = buffer1[0] / norm_factor;
    var[c] = buffer2[0] / norm_factor;
  }
}

static __global__ void kernel_forward( const int num, const int channels, const int spatial_dim, 
		const float* mean, const float* var,  const float* bottom_data,  float* top_data) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    top_data[index] = (bottom_data[index] - mean[c]) / sqrt(var[c] + BN_EPS);
  }
}
//------------------------ backward -------

static __global__ void kernel_backward_mean_var(const int num, const int channels, const int spatial_dim,
		const float* top_diff, const float* bottom_data, const float * mean_data, const float * var_data, float* mean_diff, float* var_diff) 
{
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += top_diff[index] / sqrt(var_data[c] + BN_EPS);
    buffer2[tid] += top_diff[index] * (bottom_data[index] - mean_data[c]) / sqrt(var_data[c] + BN_EPS);   
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
    mean_diff[c] = - buffer1[0];
    var_diff[c] =  - buffer2[0] / (2*(var_data[c] + BN_EPS));
  }
}

static __global__ void kernel_backward_bottom_0(const int num, const int channels, const int spatial_dim, const float norm_factor, const float* top_diff, 
			const float* var, float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    const float inv_std = float(1) / sqrt(var[c] + BN_EPS);
    bottom_diff[index]  =  inv_std * top_diff[index];    							
  }
}

static __global__ void kernel_backward_bottom_1(const int num, const int channels, const int spatial_dim, const float norm_factor, const float* top_diff, 
			const float* var, float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    const float inv_std = float(1) / sqrt(var[c] + BN_EPS);
    bottom_diff[index]  +=  inv_std * top_diff[index];    							
  }
}

static __global__ void kernel_mean_var_backward_bottom(const int num, const int channels, const int spatial_dim, const float norm_factor, 
		const float * mean_data, const float* var_data,const float * mean_diff, const float * var_diff, 
		const float* bottom_data, float* bottom_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    bottom_diff[index] += mean_diff[c] / norm_factor 
    									 + var_diff[c] / norm_factor * float(2) * (bottom_data[index] - mean_data[c]);
  }
}
//----------------------------------------secforward-----------------------------
//------------------------ diff ------------------

static __global__ void kernel_secforward_diff_mean_diff_var(const int num, const int channels, const int spatial_dim, const int norm_factor,
		const float* bottom_sec_diff, const float* bottom_data, const float * mean_data, const float * var_data, float* mean_sec_diff, float* var_sec_diff) 
{
  __shared__ float buffer1[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer2[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer1[tid] = buffer2[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer1[tid] += bottom_sec_diff[index];
    buffer2[tid] += bottom_sec_diff[index] * (bottom_data[index] - mean_data[c]);   
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
    mean_sec_diff[c] =  buffer1[0]  / norm_factor;
    var_sec_diff[c] =   buffer2[0] * float(2) / norm_factor;
  }
}

static __global__ void kernel_secforward_top(const int num, const int channels, const int spatial_dim, const float norm_factor, const float* bottom_sec_diff, 
			const float* var, float* top_sec_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    const float inv_std = float(1) / sqrt(var[c] + BN_EPS);
    top_sec_diff[index]  =  inv_std * bottom_sec_diff[index];    							
  }
}

static __global__ void kernel_diff_mean_diff_var_secforward_top(const int num, const int channels, const int spatial_dim, const float norm_factor, 
		const float * mean_data, const float* var_data,const float * mean_sec_diff, const float * var_sec_diff, 
		const float* bottom_data, float* top_sec_diff) 
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    top_sec_diff[index] += - mean_sec_diff[c] / sqrt(var_data[c]+BN_EPS)
    									 - var_sec_diff[c] * (bottom_data[index] - mean_data[c]) / pow(var_data[c]+BN_EPS, float(1.5)) * float(0.5);
  }
}

//------------------------- data --------------

static __global__ void kernel_secforward_bottom(const int num, const int channels, const int spatial_dim, const float norm_factor, 
  const float * bottom_sec_diff, const float * top_diff,
  const float * var_data, const float * var_diff, const float * var_sec_diff,
  float * bottom_diff)
{
  CUDA_KERNEL_LOOP(index, num * channels * spatial_dim) 
  {
    int c = (index / spatial_dim) % channels;
    bottom_diff[index] = bottom_sec_diff[index]*var_diff[c]*float(2)/norm_factor
    									 - var_sec_diff[c]*top_diff[index]/pow(var_data[c]+BN_EPS,float(1.5))*float(0.5); 		 
  }
}

static __global__ void kernel_secforward_mean_var(const int num, const int channels, const int spatial_dim, const float norm_factor,
		const float * bottom_sec_diff, const float * top_diff, const float * bottom_data,
		const float * mean_data, const float * mean_sec_diff, const float * var_data,	 const float * var_sec_diff,
		float * mean_diff, float * var_diff)
{
	__shared__ float buffer_secx[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer_dy[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer_secx_dy[CAFFE_CUDA_NUM_THREADS];
  __shared__ float buffer_x_dy[CAFFE_CUDA_NUM_THREADS];
  const int tid = threadIdx.x;
  const int c = blockIdx.x;

  // load and accumulate data on each thread
  buffer_secx[tid] = buffer_dy[tid] = buffer_secx_dy[tid] = buffer_x_dy[tid] = 0;
  for (int i = tid; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + c * spatial_dim + i % spatial_dim;
    buffer_secx[tid] += bottom_sec_diff[index];
    buffer_dy[tid] += top_diff[index]; 
    buffer_secx_dy[tid] += bottom_sec_diff[index]*top_diff[index];
    buffer_x_dy[tid] += (bottom_data[index] - mean_data[c])*top_diff[index];
  }
  __syncthreads();

  // do tree reduction
  for (int s = blockDim.x / 2; s > 0; s >>= 1) 
  {
    if (tid < s) 
    {
      buffer_secx[tid] += buffer_secx[tid + s];
      buffer_dy[tid] += buffer_dy[tid + s];
      buffer_secx_dy[tid] += buffer_secx_dy[tid + s];
      buffer_x_dy[tid] += buffer_x_dy[tid + s];
    }
    __syncthreads();
  }

  // save the result back
  if (tid == 0) 
  {
    mean_diff[c] = -buffer_secx[0]*var_diff[c]*float(2)/norm_factor+var_sec_diff[c]*buffer_dy[0]/pow(var_data[c]+BN_EPS,float(1.5))*float(0.5); 
    var_diff[c] = -buffer_secx_dy[0]/pow(var_data[c]+BN_EPS,float(1.5))*float(0.5)
    							+mean_sec_diff[c]*buffer_dy[0]/pow(var_data[c]+BN_EPS,float(1.5))*float(0.5)
    							+var_sec_diff[c]*buffer_x_dy[0]/pow(var_data[c]+BN_EPS,float(2.5))*float(0.75); 		 
  }
}	
//----------------------------------------------------

void BatchNormLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	if (Caffe::bn_state() == "frozen")
	{
		kernel_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		( num, channels, height * width,
			this->blobs_[0]->gpu_data(), this->blobs_[1]->gpu_data(),
			bottom[0]->gpu_data(),
			top[0]->mutable_gpu_data());
	} 	
	else
	{
		kernel_local_stats<<<channels, CAFFE_CUDA_NUM_THREADS>>>
		( num, channels, height * width,
			static_cast<float>(num * height * width),
			bottom[0]->gpu_data(),
			mean_buffer_->mutable_gpu_data(),
			var_buffer_->mutable_gpu_data());

		caffe_gpu_mul(channels, mean_buffer_->gpu_data(), mean_buffer_->gpu_data(), mean_buffer_->mutable_gpu_sec_diff()); 
		caffe_gpu_sub(channels, var_buffer_->gpu_data(), mean_buffer_->gpu_sec_diff(), var_buffer_->mutable_gpu_data());
	 	
	 	if (Caffe::number_collect_sample == 0 && Caffe::bn_state() == "learned")
		{
			caffe_gpu_set(this->blobs_[0]->count(),float(0),this->blobs_[0]->mutable_gpu_data());
			caffe_gpu_set(this->blobs_[1]->count(),float(0),this->blobs_[1]->mutable_gpu_data());
		}
		
		float factor;
		if (Caffe::number_collect_sample == -1)
			factor = 0.01;
		else 
			factor = float(1)/float(Caffe::number_collect_sample+1);
			
		caffe_gpu_axpby(mean_buffer_->count(),
	    factor, mean_buffer_->gpu_data(),
	    1-factor, this->blobs_[0]->mutable_gpu_data());
		caffe_gpu_axpby(var_buffer_->count(),
		  factor, var_buffer_->gpu_data(),
		  1-factor, this->blobs_[1]->mutable_gpu_data());

		kernel_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		( num, channels, height * width,
			mean_buffer_->gpu_data(), var_buffer_->gpu_data(),
			bottom[0]->gpu_data(),
			top[0]->mutable_gpu_data());
	}
}

void BatchNormLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  if (this->has_bottom_sec_diff_ ==  false)
	{
		kernel_backward_bottom_0<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(num, channels, height * width,  static_cast<float>(num * height * width), top[0]->gpu_diff(),var_buffer_->gpu_data(),
		bottom[0]->mutable_gpu_diff());  
	}
	else
	{
		kernel_backward_bottom_1<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
		(num, channels, height * width,  static_cast<float>(num * height * width), top[0]->gpu_diff(),var_buffer_->gpu_data(),
		bottom[0]->mutable_gpu_diff());  
	}

	kernel_backward_mean_var<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,
		top[0]->gpu_diff(), bottom[0]->gpu_data(),mean_buffer_->gpu_data(),var_buffer_->gpu_data(),
		mean_buffer_->mutable_gpu_diff(), var_buffer_->mutable_gpu_diff());  
  

	kernel_mean_var_backward_bottom<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,  static_cast<float>(num * height * width),
		mean_buffer_->gpu_data(), var_buffer_->gpu_data(),mean_buffer_->gpu_diff(), var_buffer_->gpu_diff(),
		bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
		
	this->has_bottom_sec_diff_ = false;
}


void BatchNormLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
//-------------------------------------- diff---------------------------------------
	kernel_secforward_diff_mean_diff_var<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width, static_cast<float>(num * height * width),
		bottom[0]->gpu_sec_diff(), bottom[0]->gpu_data(),mean_buffer_->gpu_data(),var_buffer_->gpu_data(),
		mean_buffer_->mutable_gpu_sec_diff(), var_buffer_->mutable_gpu_sec_diff());

	kernel_secforward_top<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
  (num, channels, height * width,  static_cast<float>(num * height * width), bottom[0]->gpu_sec_diff(),var_buffer_->gpu_data(),
  top[0]->mutable_gpu_sec_diff());       	
		
	kernel_diff_mean_diff_var_secforward_top<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,  static_cast<float>(num * height * width),
		mean_buffer_->gpu_data(), var_buffer_->gpu_data(), mean_buffer_->gpu_sec_diff(), var_buffer_->gpu_sec_diff(),
		bottom[0]->gpu_data(), top[0]->mutable_gpu_sec_diff());
	
//--------------------------------------- data ----------------------------
	kernel_secforward_bottom<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	(num, channels, height * width,  static_cast<float>(num * height * width), 
	bottom[0]->gpu_sec_diff(), top[0]->gpu_diff(),
	var_buffer_->gpu_data(), var_buffer_->gpu_diff(), var_buffer_->gpu_sec_diff(),
	bottom[0]->mutable_gpu_diff()); 
  
  kernel_secforward_mean_var<<<channels, CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width, static_cast<float>(num * height * width), 
		bottom[0]->gpu_sec_diff(),top[0]->gpu_diff(),bottom[0]->gpu_data(),
		mean_buffer_->gpu_data(), mean_buffer_->gpu_sec_diff(),var_buffer_->gpu_data(),	var_buffer_->gpu_sec_diff(),
		mean_buffer_->mutable_gpu_diff(), var_buffer_->mutable_gpu_diff());  
	
  
  kernel_mean_var_backward_bottom<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
	( num, channels, height * width,  static_cast<float>(num * height * width),
		mean_buffer_->gpu_data(), var_buffer_->gpu_data(),mean_buffer_->gpu_diff(), var_buffer_->gpu_diff(),
		bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff()); 

	this->has_bottom_sec_diff_ = true;
//---------------------------------------------------------------------------   
}

}  // namespace caffe
