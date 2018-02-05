#include <vector>

#include "caffe/layers/func/concat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
//-------------------------------------------------------

static __global__ void concat_forward(int count,int channels, int i_channels, int cur_channels,int spatial_dim,
																const float *a, float *b)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / i_channels;
		int c = i / spatial_dim % i_channels;
		int s = i % spatial_dim;
		
		b[(n*channels+cur_channels+c)*spatial_dim+s] = a[i];
	}
}

static __global__ void concat_backward(int count,int channels, int i_channels, int cur_channels,int spatial_dim,
																const float * b, float *a)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / i_channels;
		int c = i / spatial_dim % i_channels;
		int s = i % spatial_dim;
		
		a[i] = b[(n*channels+cur_channels+c)*spatial_dim+s];		
	}
}
//----------------------------------------------------

static __global__ void concat2_forward(int count,int channels,int channels0,int channels1,int spatial_dim,
																const float *a0, const float * a1, float *b)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;
		
		if (c<channels0)
			b[i] = a0[((n*channels0)+c)*spatial_dim+s];
		else 
			b[i] = a1[((n*channels1)+c-channels0)*spatial_dim+s];
		
			
	}
}

static __global__ void concat2_backward(int count,int channels,int channels0,int channels1,int spatial_dim,
																const float * b, float *a0,  float * a1)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		int s = i % spatial_dim;

		
		if (c<channels0)
			a0[((n*channels0)+c)*spatial_dim+s] = b[i];
		else
			a1[((n*channels1)+c-channels0)*spatial_dim+s] = b[i];
			
	}
}
//----------------------------------------------------

void ConcatLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (bottom.size() == 2)
	{
		int num = bottom[0]->num();
		int channels = top[0]->channels();
		int channels0 = bottom[0]->channels();
		int channels1 = bottom[1]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		concat2_forward<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[0]->count(),channels,channels0,channels1,height*width,
					bottom[0]->gpu_data(),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());
	}
	else if (bottom.size() > 2)
	{
		int num = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		
		int cur_channels = 0;
		for (int i =0; i < bottom.size();i++)
		{
			int i_channels = bottom[i]->channels();
			concat_forward<<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(bottom[i]->count(),channels,i_channels,cur_channels,height*width,
						bottom[i]->gpu_data(), top[0]->mutable_gpu_data());		
			cur_channels += i_channels;
		}
	}
	else
		LOG(FATAL)<<"wrong bottom.size";
}


void ConcatLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if  (bottom.size() == 2)
	{
		int num = bottom[0]->num();
		int channels = top[0]->channels();
		int channels0 = bottom[0]->channels();
		int channels1 = bottom[1]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		concat2_backward<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[0]->count(),channels,channels0,channels1,height*width,
					top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff(),bottom[1]->mutable_gpu_diff());		
	}
	else if (bottom.size() > 2)
	{
		int num = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		
		int cur_channels = 0;
		for (int i =0; i < bottom.size();i++)
		{
			int i_channels = bottom[i]->channels();
			concat_backward<<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(bottom[i]->count(),channels,i_channels,cur_channels,height*width,
						top[0]->gpu_diff(), bottom[i]->mutable_gpu_diff());		
			cur_channels += i_channels;
		}
	}
	else
		LOG(FATAL)<<"wrong bottom.size";
}

void ConcatLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (bottom.size() == 2)
	{
		int num = bottom[0]->num();
		int channels = top[0]->channels();
		int channels0 = bottom[0]->channels();
		int channels1 = bottom[1]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();

		concat2_forward<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(top[0]->count(),channels,channels0,channels1,height*width,
					bottom[0]->gpu_sec_diff(),bottom[1]->gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
	}
	else if (bottom.size() > 2)
	{
		int num = top[0]->num();
		int channels = top[0]->channels();
		int height = top[0]->height();
		int width = top[0]->width();
		
		int cur_channels = 0;
		for (int i =0; i < bottom.size();i++)
		{
			int i_channels = bottom[i]->channels();
			concat_forward<<<CAFFE_GET_BLOCKS(bottom[i]->count()), CAFFE_CUDA_NUM_THREADS>>>
			(bottom[i]->count(),channels,i_channels,cur_channels,height*width,
						bottom[i]->gpu_sec_diff(), top[0]->mutable_gpu_sec_diff());		
			cur_channels += i_channels;
		}
	}
	else
		LOG(FATAL)<<"wrong bottom.size";
}
}  // namespace caffe
