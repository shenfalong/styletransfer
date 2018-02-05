#ifndef CAFFE_SmoothL1Loss_LAYER_HPP_
#define CAFFE_SmoothL1Loss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class SmoothL1LossLayer : public Layer {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "SmoothL1Loss"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);


  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
 protected:


  Blob counts_;
  Blob loss_;
  
  int ignore_value;
};

}  // namespace caffe

#endif  // CAFFE_SmoothL1Loss_LAYER_HPP_
