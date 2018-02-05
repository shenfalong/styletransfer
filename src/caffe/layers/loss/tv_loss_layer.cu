
#include <vector>

#include "caffe/layers/loss/tv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/format.hpp"

namespace caffe {

static __global__ void tvloss_forward(int count, int channels, int height, int width, const float * data, float *loss)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int h = i / width % height;
		int w = i % width;
		
		float per_loss = 0;
		if (w < width - 1)
			per_loss += (data[i] - data[i+1]) * (data[i] - data[i+1]);
		
		if (h < height - 1)
			per_loss += (data[i] - data[i+width]) * (data[i] - data[i+width]);
			
		loss[i] = per_loss;	
	
	}
}


static __global__ void tvloss_backward(int count, int channels, int height, int width, const float * data, float *diff)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int h = i / width % height;
		int w = i % width;
		
		float per_diff = 0;
		if (w < width-1)
			per_diff += 2*(data[i] - data[i+1]);
		if (w > 0)
			per_diff -= 2*(data[i-1] - data[i]);
				
		if (h < height-1)			
			per_diff += 2*(data[i] - data[i+width]);
		if (h > 0)
			per_diff -= 2*(data[i-width] - data[i]);
		
		diff[i] = per_diff;	
	}
}


void TVLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	tvloss_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height,width,bottom[0]->gpu_data(),loss_.mutable_gpu_data());
	
	
	float loss;
	caffe_gpu_asum(bottom[0]->count(),loss_.gpu_data(),&loss);
	top[0]->mutable_cpu_data()[0] = loss / float(bottom[0]->count());
}


void TVLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	tvloss_backward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),channels,height,width,bottom[0]->gpu_data(),bottom[0]->mutable_gpu_diff());
	
	float loss_weight_ = top[0]->cpu_diff()[0] / float(bottom[0]->count());
	caffe_gpu_scal(bottom[0]->count(),loss_weight_,bottom[0]->mutable_gpu_diff());
}

void TVLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
