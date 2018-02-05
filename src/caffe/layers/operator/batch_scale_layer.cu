
#include <vector>

#include "caffe/layers/operator/batch_scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


static __global__ void compute_max(int num, int channels, int spatial_dim, const float *in, float *out)
{
	__shared__ float buffer[CAFFE_CUDA_NUM_THREADS];

	buffer[threadIdx.x] = float(-10000000.0);
	for (int i = threadIdx.x; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + blockIdx.x * spatial_dim + i % spatial_dim;
    buffer[threadIdx.x] = max(buffer[threadIdx.x],in[index]);
  }
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			buffer[threadIdx.x] = max(buffer[threadIdx.x], buffer[threadIdx.x+s]);
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		out[blockIdx.x] = buffer[0];
}


static __global__ void compute_min(int num, int channels, int spatial_dim, const float *in, float *out)
{
	__shared__ float buffer[CAFFE_CUDA_NUM_THREADS];

	buffer[threadIdx.x] = float(10000000.0);
	for (int i = threadIdx.x; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + blockIdx.x * spatial_dim + i % spatial_dim;
    buffer[threadIdx.x] = min(buffer[threadIdx.x],in[index]);
  }
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
			buffer[threadIdx.x] = min(buffer[threadIdx.x], buffer[threadIdx.x+s]);
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
		out[blockIdx.x] = buffer[0];
}


static __global__ void compute_sum_diff(int num, int channels, int spatial_dim, const float *diff_out, const float *in, float *out, float *out_x)
{
	__shared__ float buffer[CAFFE_CUDA_NUM_THREADS];
	__shared__ float buffer_x[CAFFE_CUDA_NUM_THREADS];
	
	buffer[threadIdx.x] = 0;
	buffer_x[threadIdx.x] = 0;
	for (int i = threadIdx.x; i < num * spatial_dim; i += blockDim.x) 
  {
    const int index = i / spatial_dim * channels * spatial_dim + blockIdx.x * spatial_dim + i % spatial_dim;
    buffer[threadIdx.x] += diff_out[index];
    buffer_x[threadIdx.x] += diff_out[index]*in[index];
  }
	__syncthreads();
	
	for (int s = blockDim.x / 2; s > 0; s >>= 1)
	{
		if (threadIdx.x < s)
		{
			buffer[threadIdx.x] += buffer[threadIdx.x+s];
			buffer_x[threadIdx.x] += buffer_x[threadIdx.x+s];
		}
		__syncthreads();
	}
	
	if (threadIdx.x == 0)
	{
		out[blockIdx.x] = buffer[0];
		out_x[blockIdx.x] = buffer_x[0];
	}
}


static __global__ void forward_kernel(int count, int channels,int spatial_dim, const float *in, const float *min_value, const float * max_value, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int c = i / spatial_dim % channels;
		out[i] = (in[i] - min_value[c]) / (max_value[c] - min_value[c]) - float(0.5);
	}
}


static __global__ void backward_kernel(int count, int channels,int spatial_dim, const float *diff_out, 
						const float *min_value, const float * max_value, const float *sum, const float *sum_x,
						const float * in,  float *diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int c = i / spatial_dim % channels;
		float gap = max_value[c] - min_value[c];
		if (in[i] != max_value[c] && in[i] != min_value[c])
			diff_in[i] = diff_out[i] / gap;
		else if (in[i] == max_value[c])
			diff_in[i] = diff_out[i] / gap +  (sum[c]*min_value[c] - sum_x[c]) / (gap*gap);
		else 
			diff_in[i] = diff_out[i] / gap +  (sum_x[c] -  sum[c]*max_value[c]) / (gap*gap);
	}
}


void BatchScaleLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_min<<<channels,CAFFE_CUDA_NUM_THREADS>>>
	(num, channels, height*width,bottom[0]->gpu_data(),this->blobs_[2]->mutable_gpu_data());
  
	compute_max<<<channels,CAFFE_CUDA_NUM_THREADS>>>
	(num, channels, height*width,bottom[0]->gpu_data(),this->blobs_[3]->mutable_gpu_data());
	
	
	forward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height*width,bottom[0]->gpu_data(),this->blobs_[2]->gpu_data(),this->blobs_[3]->gpu_data(),
			top[0]->mutable_gpu_data());
			
	
}


void BatchScaleLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_sum_diff<<<channels,CAFFE_CUDA_NUM_THREADS>>>
	(num, channels, height*width,top[0]->gpu_diff(),bottom[0]->gpu_data(),sum_.mutable_gpu_data(),sum_.mutable_gpu_diff());
  
  //LOG(ERROR)<<this->blobs_[2]->cpu_data()[0]<<", "<<this->blobs_[3]->cpu_data()[0]<<", "<<sum_.cpu_data()[0];
	//LOG(ERROR)<<bottom[0]->cpu_data()[0]<<", "<<bottom[0]->cpu_data()[1]<<", "<<bottom[0]->cpu_data()[2];
	//LOG(ERROR)<<top[0]->cpu_data()[0]<<", "<<top[0]->cpu_data()[1]<<", "<<top[0]->cpu_data()[2];
  
	backward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height*width,top[0]->gpu_diff(),this->blobs_[2]->gpu_data(),this->blobs_[3]->gpu_data(), sum_.gpu_data(),sum_.gpu_diff(),
			bottom[0]->gpu_data(), bottom[0]->mutable_gpu_diff());
	
	//LOG(ERROR)<<top[0]->cpu_diff()[0]<<", "<<top[0]->cpu_diff()[1]<<", "<<top[0]->cpu_diff()[2];
  //LOG(ERROR)<<bottom[0]->cpu_diff()[0]<<", "<<bottom[0]->cpu_diff()[1]<<", "<<bottom[0]->cpu_diff()[2];

}


void BatchScaleLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
