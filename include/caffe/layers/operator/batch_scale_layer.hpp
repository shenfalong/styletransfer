
#ifndef CAFFE_BatchScale_LAYER_HPP_
#define CAFFE_BatchScale_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class BatchScaleLayer : public Layer {
 public:
  explicit BatchScaleLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "BatchScale"; }
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:

	Blob sum_;
 	
};

}  // namespace caffe

#endif  // CAFFE_BatchScaleLAYER_HPP_
