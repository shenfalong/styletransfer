#ifndef CAFFE_PARSE_OUTPUT_LAYER_HPP_
#define CAFFE_PARSE_OUTPUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


class ParseOutputLayer : public Layer 
{
 public:
  explicit ParseOutputLayer(const LayerParameter& param) : Layer(param) {}
  virtual inline const char* type() const { return "ParseOutput"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);


	
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
  bool out_max_val_;

  // max_prob_ is used to store the maximum probability value
  Blob max_prob_;
};
}

#endif  // CAFFE_LOSS_LAYER_HPP_
