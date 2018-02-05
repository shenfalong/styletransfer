#include <vector>

#include "caffe/layers/activation/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



static __global__ void DropoutForward(const int count, int channels, int spatial, const float threshold, const float* in, const float* rand_vec, float* out) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
  	int c = index / spatial % channels;
  	if (rand_vec[c] > threshold)
    	out[index] = in[index];
    else
    	out[index] = 0;
  }
}


static __global__ void DropoutBackward(const int count, int channels, int spatial, const float threshold, const float* in_diff, const float * rand_vec, float* out_diff) 
{
  CUDA_KERNEL_LOOP(index, count) 
  {
  	int c = index / spatial % channels;
  	if (rand_vec[c] > threshold)
    	out_diff[index] = in_diff[index];
   	else
    	out_diff[index] = 0;
  }
}


void DropoutLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();	
	
  if (Caffe::drop_state() == "rand")
  {
    caffe_gpu_rng_uniform(channels,float(0.0),float(1.0), rand_vec_.mutable_gpu_data());
    

    DropoutForward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
    (bottom[0]->count(),channels,height*width,threshold_, bottom[0]->gpu_data(), rand_vec_.gpu_data(), top[0]->mutable_gpu_data());
  } 
  else 
  {
    caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
    caffe_gpu_scal(top[0]->count(),float(1.0)-threshold_,top[0]->mutable_gpu_data());
  }
  
}


void DropoutLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();	
  
  
  DropoutBackward<<<CAFFE_GET_BLOCKS(bottom[0]->count()),CAFFE_CUDA_NUM_THREADS>>>
  (bottom[0]->count(),channels,height*width, threshold_, top[0]->gpu_diff(), rand_vec_.gpu_data(), bottom[0]->mutable_gpu_diff());
}


void DropoutLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	
}



}  // namespace caffe
