#include <cfloat>
#include <vector>

#include "caffe/layers/func/eltwise_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


static __global__ void MaxForward(const int nthreads, const float* bottom_data_a,
    const float* bottom_data_b, const int blob_idx, float* top_data,
    float* mask) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    float maxval = -FLT_MAX;
    int maxidx = -1;
    if (bottom_data_a[index] > bottom_data_b[index]) {
      // only update for very first bottom_data blob (blob_idx == 0)
      if (blob_idx == 0) {
        maxval = bottom_data_a[index];
        top_data[index] = maxval;
        maxidx = blob_idx;
        mask[index] = maxidx;
      }
    } else {
      maxval = bottom_data_b[index];
      top_data[index] = maxval;
      maxidx = blob_idx + 1;
      mask[index] = maxidx;
    }
  }
}


void EltwiseLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{ 
  float * mask = NULL;
  const int count = top[0]->count();
  float* top_data = top[0]->mutable_gpu_data();
  if (op_ == "prod")
 	{
    caffe_gpu_mul(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        top_data);
    for (int i = 2; i < bottom.size(); ++i) {
      caffe_gpu_mul(count, top_data, bottom[i]->gpu_data(), top_data);
    }
  }
  else if (op_ == "sum")
  {
    caffe_gpu_set(count, float(0.), top_data);
    // TODO(shelhamer) does cuBLAS optimize to sum for coeff = 1?
    for (int i = 0; i < bottom.size(); ++i) {
      caffe_gpu_axpy(count, coeffs_[i], bottom[i]->gpu_data(), top_data);
    }
  }
  else if (op_ == "max")
  {
    mask = max_idx_.mutable_gpu_data();
    // NOLINT_NEXT_LINE(whitespace/operators)
    MaxForward <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, bottom[0]->gpu_data(), bottom[1]->gpu_data(), 0, top_data, mask);
    for (int i = 2; i < bottom.size(); ++i) {
      // NOLINT_NEXT_LINE(whitespace/operators)
      MaxForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_data, bottom[i]->gpu_data(), i-1, top_data, mask);
    }
  }
  else
    LOG(FATAL) << "Unknown elementwise operation.";
}


__global__ void MaxBackward(const int nthreads, const float* top_diff,
    const int blob_idx, const float* mask, float* bottom_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    float gradient = 0;
    if (mask[index] == blob_idx) {
      gradient += top_diff[index];
    }
    bottom_diff[index] = gradient;
  }
}


void EltwiseLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
    
  const float* mask = NULL;
  const int count = top[0]->count();
  const float* top_data = top[0]->gpu_data();
  const float* top_diff = top[0]->gpu_diff();
  for (int i = 0; i < bottom.size(); ++i) 
  {
    const float* bottom_data = bottom[i]->gpu_data();
    float* bottom_diff = bottom[i]->mutable_gpu_diff();
    if (op_ == "prod")
    {
      if (stable_prod_grad_) {
        bool initialized = false;
        for (int j = 0; j < bottom.size(); ++j) {
          if (i == j) { continue; }
          if (!initialized) {
            caffe_copy(count, bottom[j]->gpu_data(), bottom_diff);
            initialized = true;
          } else {
            caffe_gpu_mul(count, bottom[j]->gpu_data(), bottom_diff,
                          bottom_diff);
          }
        }
      } else {
        caffe_gpu_div(count, top_data, bottom_data, bottom_diff);
      }
      caffe_gpu_mul(count, bottom_diff, top_diff, bottom_diff);
    }
    else if (op_ == "sum")
    {
      if (coeffs_[i] == float(1.)) {
        caffe_copy(count, top_diff, bottom_diff);
      } else {
        caffe_gpu_scale(count, coeffs_[i], top_diff, bottom_diff);
      }
    }
    else if (op_ == "max")
    {
      mask = max_idx_.gpu_data();
      MaxBackward  // NOLINT_NEXT_LINE(whitespace/operators)
          <<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
          count, top_diff, i, mask, bottom_diff);
    }
    else
      LOG(FATAL) << "Unknown elementwise operation.";
    
  	if (backwards_[i] == false)
  		caffe_gpu_set(bottom[i]->count(),float(0),bottom[i]->mutable_gpu_diff());
  }
  
}

void EltwiseLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{ 
}


}  // namespace caffe
