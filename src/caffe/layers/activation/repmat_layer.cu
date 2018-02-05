
#include <vector>

#include "caffe/layers/activation/repmat_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void repmat_kernel(int count, int channels,int spatial_dim, const float *in, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim / channels;
		int c = i / spatial_dim % channels;
		out[i] = in[n*channels+c];
	}
}


void RepmatLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim;
  if (bottom.size() == 2)
  	spatial_dim = bottom[1]->height() * bottom[1]->width();
	else
		spatial_dim = this->layer_param_.shape_param().height() * this->layer_param_.shape_param().width();
	
	repmat_kernel<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(top[0]->count(),channels,spatial_dim,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}


void RepmatLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int spatial_dim;
  if (bottom.size() == 2)
  	spatial_dim = bottom[1]->height() * bottom[1]->width();
	else
		spatial_dim = this->layer_param_.shape_param().height() * this->layer_param_.shape_param().width();
  
  caffe_gpu_set(one_multiplier_->count(),float(1),one_multiplier_->mutable_gpu_data());
  
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num*channels, 1, spatial_dim,
														(float)1., top[0]->gpu_diff() , one_multiplier_->gpu_data(),
														(float)0., bottom[0]->mutable_gpu_diff());  
}

void RepmatLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

}

}  // namespace caffe
