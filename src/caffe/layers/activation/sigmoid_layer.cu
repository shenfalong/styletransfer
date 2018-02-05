#include <cmath>
#include <vector>

#include "caffe/layers/activation/sigmoid_layer.hpp"

namespace caffe {


static __global__ void SigmoidForward(const int n, const float* in, float* out) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    out[index] = 1. / (1. + exp(-in[index]));
  }
}


void SigmoidLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
}


static __global__ void SigmoidBackward(const int n, const float* in_diff, const float* out_data, float* out_diff) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    const float sigmoid_x = out_data[index];
    out_diff[index] = in_diff[index] * sigmoid_x * (1 - sigmoid_x);
  }
}


void SigmoidLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
  const float* top_data = top[0]->gpu_data();
  const float* top_diff = top[0]->gpu_diff();
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = bottom[0]->count();
  // NOLINT_NEXT_LINE(whitespace/operators)
  SigmoidBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, top_diff, top_data, bottom_diff);
  CUDA_POST_KERNEL_CHECK;
}

void SigmoidLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}




}  // namespace caffe
