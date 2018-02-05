
#ifndef CAFFE_Repmat_LAYER_HPP_
#define CAFFE_Repmat_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class RepmatLayer : public Layer {
 public:
  explicit RepmatLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "Repmat"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	int gpu_id_;
 	Blob * one_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_RepmatLAYER_HPP_
