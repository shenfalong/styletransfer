#ifndef CAFFE_PARSE_EVALUATE_LAYER_HPP_
#define CAFFE_PARSE_EVALUATE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


class ParseEvaluateLayer : public Layer {
 public:
  explicit ParseEvaluateLayer(const LayerParameter& param) : Layer(param) {}
  virtual inline const char* type() const { return "ParseEvaluate"; }
  
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);


	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:

  // number of total labels
  int num_labels_;
  // store ignored labels
  std::set<float> ignore_labels_;
};

}

#endif  // CAFFE_LOSS_LAYER_HPP_
