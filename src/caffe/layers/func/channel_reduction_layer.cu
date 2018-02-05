
#include <vector>

#include "caffe/layers/func/channel_reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void forward_kernel(int count, int channels,int spatial_dim, const float *in, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;

		out[i] = max(in[(n*channels*2+c)*spatial_dim+s],in[(n*channels*2+channels+c)*spatial_dim+s]);
	}
}

static __global__ void backward_kernel(int count, int channels,int spatial_dim, const float * diff_out, const float *in, float *diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		if (in[(n*channels*2+c)*spatial_dim+s] > in[(n*channels*2+channels+c)*spatial_dim+s])
		{
			diff_in[(n*channels*2+c)*spatial_dim+s] = diff_out[i];
			diff_in[(n*channels*2+channels+c)*spatial_dim+s] = float(0);
		}
		else
		{
			diff_in[(n*channels*2+c)*spatial_dim+s] = float(0);
			diff_in[(n*channels*2+channels+c)*spatial_dim+s] = diff_out[i];
		}
	}
}


void ChannelReductionLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = top[0]->num();
  int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
	
	forward_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}


void ChannelReductionLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = top[0]->num();
  int channels = top[0]->channels();
  int height = top[0]->height();
  int width = top[0]->width();
  
	backward_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,height*width,top[0]->gpu_diff(),bottom[0]->gpu_data(),bottom[0]->mutable_gpu_diff());
}


void ChannelReductionLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
