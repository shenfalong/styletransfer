#ifndef CAFFE_SPLIT_LAYER_HPP_
#define CAFFE_SPLIT_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Creates a "split" path in the network by copying the bottom Blob
 *        into multiple top Blob%s to be used by multiple consuming layers.
 *
 * TODO(dox): thorough documentation for Forward, Backward, and proto params.
 */

class SplitLayer : public Layer 
{
 public:
  explicit SplitLayer(const LayerParameter& param) : Layer(param) {}
  virtual inline const char* type() const { return "Split"; }
  
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
	virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

 protected:


};

}  // namespace caffe

#endif  // CAFFE__SPLIT_LAYER_HPP_
