#ifndef CAFFE_ADAPT_SOFTMAX_WITH_LOSS_LAYER_HPP_
#define CAFFE_ADAPT_SOFTMAX_WITH_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/activation/softmax_layer.hpp"

namespace caffe {


class AdaptSoftmaxWithLossLayer : public Layer 
{
 public:
  explicit AdaptSoftmaxWithLossLayer(const LayerParameter& param) : Layer(param) {}
  virtual inline const char* type() const { return "AdaptSoftmaxWithLoss"; }
  
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 

  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

 protected:
  



  Blob prob_;
  
  Blob temp_prob;
  
  shared_ptr<Layer > softmax_layer_;
 
  
  Blob flag;
  vector<Blob*> softmax_bottom_vec_;
  vector<Blob*> softmax_top_vec_;

  bool has_ignore_label_;
 
  int ignore_label_;
 
  Blob counts_;
  Blob sub_counts_;
  Blob loss_;
 	float portion;
 

};

}  // namespace caffe

#endif  // CAFFE_Adapt_SOFTMAX_WITH_LOSS_LAYER_HPP_
