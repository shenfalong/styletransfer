
#include <vector>

#include "caffe/layers/activation/tanh_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


static __global__ void TanHForward(const int n, const float* in, float* out) {
  CUDA_KERNEL_LOOP(index, n) {
    out[index] = tanh(in[index]);
  }
}


static __global__ void TanHBackward(const int n, const float* in_diff,
    const float* out_data, float* out_diff) {
  CUDA_KERNEL_LOOP(index, n) {
    float tanhx = out_data[index];
    out_diff[index] = in_diff[index] * (1 - tanhx * tanhx);
  }
}

void TanHLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  TanHForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, bottom_data, top_data);
}


void TanHLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{

  const float* top_data = top[0]->gpu_data();
  const float* top_diff = top[0]->gpu_diff();
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  TanHBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, top_diff, top_data, bottom_diff);
}

void TanHLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}
;
}  // namespace caffe
