
#ifndef CAFFE_StyleLoss_LAYER_HPP_
#define CAFFE_StyleLoss_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class StyleLossLayer : public Layer {
 public:
  explicit StyleLossLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "StyleLoss"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);



  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	Blob * buffer_0_;
 	Blob * buffer_1_;
 	Blob * buffer_delta_;
 	Blob * buffer_square_;
};

}  // namespace caffe

#endif  // CAFFE_StyleLossLAYER_HPP_
