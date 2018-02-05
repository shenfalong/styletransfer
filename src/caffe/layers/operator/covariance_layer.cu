
#include <vector>

#include "caffe/layers/operator/covariance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void kernel(int count, int channels,int height,int width, const float *in, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / width / height / channels;
		int c = i / width / height % channels;
		int h = i / width % height;
		int w = i % width;
		out[i] = in[i];
	}
}


void CovarianceLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, channels, channels, height*width,
														(float)1., bottom[0]->gpu_data() + bottom[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(float)0., top[0]->mutable_gpu_data() + top[0]->offset(n));		
	}
	caffe_gpu_scal(top[0]->count(),float(1)/float(height*width),top[0]->mutable_gpu_data());	
	

}


void CovarianceLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  

  caffe_gpu_scal(top[0]->count(),float(1)/float(height*width),top[0]->mutable_gpu_diff());		
	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels, height*width, channels,
														(float)1., top[0]->gpu_diff() + top[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(float)0., bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));		
		caffe_gpu_gemm(CblasTrans, CblasNoTrans, channels, height*width, channels,
														(float)1., top[0]->gpu_diff() + top[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(float)1., bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));	
	}

}

void CovarianceLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
