
#include <vector>

#include "caffe/layers/func/rgb_gray_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void forward_kernel(int count, int spatial_dim, const float *in, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		//Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
		out[i] = float(0.11400) * in[(n*3+0)*spatial_dim+s]  
					 + float(0.58700) * in[(n*3+1)*spatial_dim+s]
					 + float(0.29900) * in[(n*3+2)*spatial_dim+s];
	}
}

static __global__ void backward_kernel(int count, int spatial_dim, const float *diff_out, float *diff_in)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		//Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
		diff_in[(n*3+0)*spatial_dim+s] = float(0.11400) * diff_out[i];
		diff_in[(n*3+1)*spatial_dim+s] = float(0.58700) * diff_out[i];
		diff_in[(n*3+2)*spatial_dim+s] = float(0.29900) * diff_out[i];
	}
}

void RGBGRAYLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	forward_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),height*width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
	//LOG(ERROR)<<bottom[0]->cpu_data()[0]<<", "<<bottom[0]->cpu_data()[1]<<", "<<bottom[0]->cpu_data()[2];
	//LOG(ERROR)<<top[0]->cpu_data()[0];
}


void RGBGRAYLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	backward_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),height*width,top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
	//LOG(ERROR)<<top[0]->cpu_diff()[0];
	//LOG(ERROR)<<bottom[0]->cpu_diff()[0]<<", "<<bottom[0]->cpu_diff()[1]<<", "<<bottom[0]->cpu_diff()[2];
	
}


void RGBGRAYLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
