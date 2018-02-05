#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/layers/activation/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


static __global__ void kernel_channel_max(const int num, const int channels,
    const int spatial_dim, const float* data, float* out) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float maxval = -FLT_MAX;
    for (int c = 0; c < channels; ++c) {
      maxval = max(data[(n * channels + c) * spatial_dim + s], maxval);
    }
    out[index] = maxval;
  }
}


static __global__ void kernel_channel_subtract(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* channel_max, float* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] -= channel_max[n * spatial_dim + s];
  }
}


static __global__ void kernel_exp(const int count, const float* data, float* out) {
  CUDA_KERNEL_LOOP(index, count) {
    out[index] = exp(data[index]);
  }
}


static __global__ void kernel_channel_sum(const int num, const int channels,
    const int spatial_dim, const float* data, float* channel_sum) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float sum = 0;
    for (int c = 0; c < channels; ++c) {
      sum += data[(n * channels + c) * spatial_dim + s];
    }
    channel_sum[index] = sum;
  }
}


static __global__ void kernel_channel_div(const int count,
    const int num, const int channels,
    const int spatial_dim, const float* channel_sum, float* data) {
  CUDA_KERNEL_LOOP(index, count) {
    int n = index / channels / spatial_dim;
    int s = index % spatial_dim;
    data[index] /= channel_sum[n * spatial_dim + s];
  }
}


static __global__ void kernel_channel_dot(const int num, const int channels,
    const int spatial_dim, const float* data_1, const float* data_2,
    float* channel_dot) {
  CUDA_KERNEL_LOOP(index, num * spatial_dim) {
    int n = index / spatial_dim;
    int s = index % spatial_dim;
    float dot = 0;
    for (int c = 0; c < channels; ++c) {
      dot += (data_1[(n * channels + c) * spatial_dim + s]
          * data_2[(n * channels + c) * spatial_dim + s]);
    }
    channel_dot[index] = dot;
  }
}


void SoftmaxLayer::Forward_gpu(const vector<Blob*>& bottom,  const vector<Blob*>& top) 
{
  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  float* scale_data = scale_.mutable_gpu_data();
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  int count = bottom[0]->count();
  
  
  caffe_copy(count, bottom_data, top_data);

  kernel_channel_max<<<CAFFE_GET_BLOCKS(num * height * width),CAFFE_CUDA_NUM_THREADS>>>
  (num, channels, height*width, top_data,scale_data);
  
  kernel_channel_subtract<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>
  (count, num, channels, height*width, scale_data, top_data);
  
  kernel_exp<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>
  (count, top_data, top_data);
  

  kernel_channel_sum<<<CAFFE_GET_BLOCKS(num * height * width),CAFFE_CUDA_NUM_THREADS>>>
  (num, channels, height*width, top_data,scale_data);
  
  kernel_channel_div<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>
  (count, num, channels, height*width, scale_data, top_data);   
}


void SoftmaxLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
  const float* top_diff = top[0]->gpu_diff();
  const float* top_data = top[0]->gpu_data();
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  float* scale_data = scale_.mutable_gpu_data();
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  int count = bottom[0]->count();
 
  caffe_copy(count, top_diff, bottom_diff);

  kernel_channel_dot<<<CAFFE_GET_BLOCKS(num * height * width),CAFFE_CUDA_NUM_THREADS>>>
  (num, channels, height*width, top_diff, top_data, scale_data);

  kernel_channel_subtract<<<CAFFE_GET_BLOCKS(count),CAFFE_CUDA_NUM_THREADS>>>
  (count, num, channels, height*width, scale_data, bottom_diff);
  
  caffe_gpu_mul(top[0]->count(), bottom_diff, top_data, bottom_diff);
}

void SoftmaxLayer::SecForward_gpu(const vector<Blob*>& bottom,  const vector<Blob*>& top) 
{

}



}  // namespace caffe
