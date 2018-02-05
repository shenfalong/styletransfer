
#ifndef CAFFE_EuclideanLoss_LAYER_HPP_
#define CAFFE_EuclideanLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class EuclideanLossLayer : public Layer {
 public:
  explicit EuclideanLossLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "EuclideanLoss"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);



  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 
 	Blob * buffer_delta_;
 	Blob * buffer_square_;
 	
};

}  // namespace caffe

#endif  // CAFFE_EuclideanLossLAYER_HPP_
