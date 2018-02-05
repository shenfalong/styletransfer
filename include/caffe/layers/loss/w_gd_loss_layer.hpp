
#ifndef CAFFE_WGdLoss_LAYER_HPP_
#define CAFFE_WGdLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class WGdLossLayer : public Layer {
 public:
  explicit WGdLossLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "WGdLoss"; }
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	Blob loss_;
 
 	Blob loss_g_;
 	Blob loss_d_;
 	Blob mask_;
 	Blob count_;
 	
};

}  // namespace caffe

#endif  // CAFFE_WGdLossLAYER_HPP_
