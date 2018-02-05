#ifndef CAFFE_POOLING_LAYER_HPP_
#define CAFFE_POOLING_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include <algorithm>
#include <cfloat>
#include <vector>

namespace caffe {


class PoolingLayer : public Layer 
{
 public:
  explicit PoolingLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "Pooling"; }
 
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
 	virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:

  int kernel_size_;
  int pad_;
  int stride_;
  int pooled_height_, pooled_width_;
  Blob max_idx_;
};

}  // namespace caffe

#endif  // CAFFE_POOLING_LAYER_HPP_
