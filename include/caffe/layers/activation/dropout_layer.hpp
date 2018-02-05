#ifndef CAFFE_DROPOUT_LAYER_HPP_
#define CAFFE_DROPOUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class DropoutLayer : public Layer {
 public:
  explicit DropoutLayer(const LayerParameter& param) : Layer(param) {}
  virtual inline const char* type() const { return "Dropout"; }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:


  Blob rand_vec_;
  Blob flag_vec_;
  float threshold_;
};

}  // namespace caffe

#endif  // CAFFE_DROPOUT_LAYER_HPP_
