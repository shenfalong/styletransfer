
#include <vector>

#include "caffe/layers/loss/max_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void forward_kernel(int count, int channels,int spatial_dim, const float *pred_data, const float *label_data, float *loss_data)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		int label = label_data[i];
		float sum = 0;
		for (int c=0;c<channels;c++)
		{
			if (c != label)
				sum += abs(pred_data[(n*channels+c)*spatial_dim+s]);
			else
				sum += abs(1 - pred_data[(n*channels+c)*spatial_dim+s]);
		}
		loss_data[i] = sum;
	}
}

static __global__ void backward_kernel(int count, int channels,int spatial_dim, const float *pred_data, const float *label_data, float *pred_diff)
{
	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		int label = label_data[i];
		for (int c=0;c<channels;c++)
		{
			if (c != label)
				pred_diff[(n*channels+c)*spatial_dim+s] = pred_data[(n*channels+c)*spatial_dim+s]>0? 1:-1;
			else
				pred_diff[(n*channels+c)*spatial_dim+s] = pred_data[(n*channels+c)*spatial_dim+s]-1>0? 1:-1;
		}	
	}
}

void MaxLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	forward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
	(num*height*width,channels,height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),loss_.mutable_gpu_data());
	
	float loss;
  caffe_gpu_asum(num*height*width, loss_.gpu_data(), &loss);
  
  top[0]->mutable_cpu_data()[0] = loss / float(num*height*width);
}


void MaxLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  if (Caffe::second_pass() == false)
	{
		backward_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		(num*height*width,channels,height*width,bottom[0]->gpu_data(),bottom[1]->gpu_data(),bottom[0]->mutable_gpu_diff());
		
		const float loss_weight = top[0]->cpu_diff()[0] / float(num*height*width);
		caffe_gpu_scal(bottom[0]->count(), loss_weight, bottom[0]->mutable_gpu_diff());
	}
}


void MaxLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
