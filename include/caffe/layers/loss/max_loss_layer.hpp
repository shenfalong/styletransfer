
#ifndef CAFFE_MaxLoss_LAYER_HPP_
#define CAFFE_MaxLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class MaxLossLayer : public Layer {
 public:
  explicit MaxLossLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "MaxLoss"; }
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	Blob loss_;
};

}  // namespace caffe

#endif  // CAFFE_MaxLossLAYER_HPP_
