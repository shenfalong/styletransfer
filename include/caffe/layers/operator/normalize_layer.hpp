#ifndef CAFFE_NORMALIZATION_LAYER_HPP_
#define CAFFE_NORMALIZATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/filler.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {


class NormalizeLayer : public Layer 
{
 public:
  explicit NormalizeLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "Normalize"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom,  const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom,     const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
	virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:


  Blob norm_;
  Blob sum_channel_multiplier_, sum_spatial_multiplier_;
  Blob buffer_, buffer_channel_, buffer_spatial_;
  float eps_;
};

}  // namespace caffe

#endif  // CAFFE_BATCHNORM_LAYER_HPP_
