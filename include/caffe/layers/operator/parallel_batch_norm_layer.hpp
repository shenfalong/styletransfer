#ifndef CAFFE_PARALLEL_BATCH_NORM_LAYER_HPP_
#define CAFFE_PARALLEL_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {




class ParallelBatchNormLayer : public Layer 
{
 public:
  explicit ParallelBatchNormLayer(const LayerParameter& param) : Layer(param) {}
  virtual ~ParallelBatchNormLayer();
  virtual inline const char* type() const { return "ParallelBatchNorm"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom,  const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
	virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
	vector<Blob *> parallel_mean_buffer_;
	vector<Blob *> parallel_var_buffer_;  
};		

}  // namespace caffe

#endif 
