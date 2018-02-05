
#ifndef CAFFE_Reshape_LAYER_HPP_
#define CAFFE_Reshape_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class ReshapeLayer : public Layer {
 public:
  explicit ReshapeLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "Reshape"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	int num_;
 	int channels_;
 	int height_;
 	int width_;
 	
};

}  // namespace caffe

#endif  // CAFFE_ReshapeLAYER_HPP_
