#include <algorithm>
#include <vector>

#include "caffe/layers/activation/relu_layer.hpp"

namespace caffe {


static __global__ void ReLUForward(const int n, const int negative_slope, const float* in, bool* flag, float* out) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
  	flag[index] = in[index] > 0;
    out[index] = in[index] > 0 ? in[index] : in[index] * negative_slope;
  }
}

static __global__ void ReLUBackward(const int n, const int negative_slope, const float* in_diff,const bool* flag, float* out_diff) 
{
  CUDA_KERNEL_LOOP(index, n)
  {
  	if (flag[index])
    	out_diff[index] = in_diff[index];
    else
    	out_diff[index] = in_diff[index] * negative_slope;
  }
}




void ReLULayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();

	bool* flag_data = reinterpret_cast<bool*>(flag.mutable_gpu_data());
	ReLUForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	(count, negative_slope, bottom_data, flag_data,top_data);
	
	//ReLUForward_test<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	//(count, negative_slope, bottom_data,top_data);
	//CUDA_POST_KERNEL_CHECK; 
	
	//CUDA_CHECK(cudaDeviceSynchronize());  
}


void ReLULayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{  
  const float* top_diff = top[0]->gpu_diff();
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  
  
	const bool* flag_data = reinterpret_cast<const bool*>(flag.gpu_data());
  ReLUBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, negative_slope, top_diff, flag_data, bottom_diff);     
}


void ReLULayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
   
 
	const bool* flag_data = reinterpret_cast<const bool*>(flag.gpu_data());
	ReLUBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
	(count, negative_slope, bottom[0]->gpu_sec_diff(), flag_data, top[0]->mutable_gpu_sec_diff());
	CUDA_POST_KERNEL_CHECK;    	
}


}  // namespace caffe
