#ifndef CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
#define CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/activation/softmax_layer.hpp"

namespace caffe {


class SoftmaxWithLossLayer : public Layer {
 public:
  explicit SoftmaxWithLossLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "SoftmaxWithLoss"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);


  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
  
 protected:

  Blob prob_;
  
  
  shared_ptr<Layer > softmax_layer_;
 
  
  
  vector<Blob*> softmax_bottom_vec_;
  vector<Blob*> softmax_top_vec_;

  bool has_ignore_label_;
 
  int ignore_label_;
 
  Blob counts_;
  Blob loss_;
};

}  // namespace caffe

#endif  // CAFFE_SOFTMAX_WITH_LOSS_LAYER_HPP_
