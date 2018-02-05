#ifndef CAFFE_RELU_LAYER_HPP_
#define CAFFE_RELU_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class ReLULayer : public Layer 
{
 public:

  explicit ReLULayer(const LayerParameter& param): Layer(param) {}
      
	
	
  virtual inline const char* type() const { return "ReLU"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 
  
  float negative_slope;

 	int gpu_id_;
  
  Blob flag;
};

}  // namespace caffe

#endif  // CAFFE_RELU_LAYER_HPP_
