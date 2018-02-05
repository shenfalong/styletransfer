#ifndef CAFFE_SHORTCUT_LAYER_HPP_
#define CAFFE_SHORTCUT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class ShortcutLayer : public Layer {
 public:
  explicit ShortcutLayer(const LayerParameter& param): Layer(param) {}
  virtual ~ShortcutLayer();
  virtual inline const char* type() const { return "Shortcut"; }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
	virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:

  
};

}  // namespace caffe

#endif  
