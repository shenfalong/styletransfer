
#ifndef CAFFE_OneHot_LAYER_HPP_
#define CAFFE_OneHot_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class OneHotLayer : public Layer {
 public:
  explicit OneHotLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "OneHot"; }
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 
 	int classes_;
 	
};

}  // namespace caffe

#endif  // CAFFE_OneHotLAYER_HPP_
