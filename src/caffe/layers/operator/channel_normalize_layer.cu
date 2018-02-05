#include <vector>
#include "caffe/layers/operator/channel_normalize_layer.hpp"
#define BN_EPS float(1e-5)

namespace caffe {
//---------------------------------------------------
#if 0

static __global__ void compute_norm(int count, int channels, const float *in, float *norm)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		float sum = 0;
		for (int c=0; c<channels; c++)
			sum += in[(n*channels+c)*spatial_dim+s]*in[(n*channels+c)*spatial_dim+s];
		sum /= float(channels);	
		norm[i] = sqrt(sum+BN_EPS);
	}
}

static __global__ void forward_kernel(int count, int channels, const float *in, const float *norm, float * out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		out[i] = float(0.02) * in[i] / norm[n*spatial_dim + s];
	}
}
//---------------------------------------------------

static __global__ void compute_diff_norm(int count, int channels, const float *diff_out, const float *in, const float *norm, float *diff_norm)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		float sum = 0;
		for (int c=0; c<channels; c++)
			sum += diff_out[(n*channels+c)*spatial_dim+s]*(-in[(n*channels+c)*spatial_dim+s])/(norm[i]*norm[i]);
		diff_norm[i] = float(0.02) * sum;
	}
}

static __global__ void backward_kernel(int count, int channels, const float *diff_out, const float * in, 
																const float *norm, const float *diff_norm, float * diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		diff_in[i] = float(0.02) * diff_out[i]/norm[n*spatial_dim+s] + diff_norm[n*spatial_dim+s]*in[i]/norm[n*spatial_dim+s]/float(channels);
	}
}
#endif
//--------------------------------------------------

void ChannelNormalizeLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
#if 0
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	compute_norm<<<num, CAFFE_CUDA_NUM_THREADS>>>
	(channels,bottom[0]->gpu_data(),norm_.mutable_gpu_data());
	
	forward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,bottom[0]->gpu_data(),norm_.gpu_data(),top[0]->mutable_gpu_data());
#endif	
}


void ChannelNormalizeLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
#if 0
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_diff_norm<<<num, CAFFE_CUDA_NUM_THREADS>>>
	(channels, top[0]->gpu_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.mutable_gpu_diff());
  
  backward_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,top[0]->gpu_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.gpu_diff(),bottom[0]->mutable_gpu_diff());
#endif	
}

void ChannelNormalizeLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
#if 0
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  compute_diff_norm<<<num, CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels, bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.mutable_gpu_sec_diff());
  
  backward_kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels, bottom[0]->gpu_sec_diff(),bottom[0]->gpu_data(),norm_.gpu_data(),norm_.gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
#endif	
}

}  // namespace caffe
