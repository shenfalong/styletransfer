
#ifndef CAFFE_Be_GdLoss_LAYER_HPP_
#define CAFFE_Be_GdLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class BeGdLossLayer : public Layer {
 public:
  explicit BeGdLossLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "BeGdLoss"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);



  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	Blob loss_d_;
 	Blob loss_g_;
 	float sum_d;
 	float sum_g;
 	
};

}  // namespace caffe

#endif  // CAFFE_BeGdLossLAYER_HPP_
