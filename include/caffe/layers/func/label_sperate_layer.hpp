#ifndef CAFFE_LabelSperate_LAYER_HPP_
#define CAFFE_LabelSperate_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class LabelSperateLayer : public Layer {
 public:
  explicit LabelSperateLayer(const LayerParameter& param): Layer(param) {}
  

  virtual inline const char* type() const { return "LabelSperate"; }
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
#if 0
 	int stuff_mapping[150];
 	int object_mapping[150];
#endif 	
};

}  // namespace caffe

#endif  // CAFFE_LabelSperate_LAYER_HPP_
		
