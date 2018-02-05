
#ifndef CAFFE_Lambda_LAYER_HPP_
#define CAFFE_Lambda_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class LambdaLayer : public Layer{
 public:
  explicit LambdaLayer(const LayerParameter& param): Layer(param) {}
  virtual inline const char* type() const { return "Lambda"; }
	virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	
 	int gpu_id_;
 	vector<shared_ptr<Layer> > all_layer_;
 	vector< vector<Blob* > > unary_bottom_vec_;
 	int layer_index_;
};

}  // namespace caffe

#endif  // CAFFE_LambdaLAYER_HPP_
