
#ifndef CAFFE_GANMultiSoftmaxWithLoss_LAYER_HPP_
#define CAFFE_GANMultiSoftmaxWithLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class GANMultiSoftmaxWithLossLayer : public Layer {
 public:
  explicit GANMultiSoftmaxWithLossLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "GANMultiSoftmaxWithLoss"; }
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	Blob loss_;
 	
 	Blob prob_;
  
  
  shared_ptr<Layer > softmax_layer_;
 
  
  
  vector<Blob*> softmax_bottom_vec_;
  vector<Blob*> softmax_top_vec_;
};

}  // namespace caffe

#endif  // CAFFE_GANMultiSoftmaxWithLossLAYER_HPP_
