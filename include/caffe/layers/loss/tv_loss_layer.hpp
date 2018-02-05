
#ifndef CAFFE_TVLoss_LAYER_HPP_
#define CAFFE_TVLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class TVLossLayer : public Layer {
 public:
  explicit TVLossLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "TVLoss"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);



  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	Blob loss_;
 	
 	int gpu_id_;
};

}  // namespace caffe

#endif  // CAFFE_TVLossLAYER_HPP_
