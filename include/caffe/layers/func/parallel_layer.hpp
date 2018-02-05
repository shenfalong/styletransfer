#ifndef CAFFE_PARALLEL_LAYER_HPP_
#define CAFFE_PARALLEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {


class ParallelLayer : public Layer 
{
 public:
  explicit ParallelLayer(const LayerParameter& param) : Layer(param)  {}
  virtual ~ParallelLayer();
  virtual inline const char* type() const { return "Parallel"; }
  
  virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Forward_gpu(const vector<Blob*>& bottom,  const vector<Blob*>& top);
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
	virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
//-------------------------------------------  
	vector<shared_ptr<Layer > > unary_layer_;
	vector< vector<Blob* > > unary_bottom_vec_;
	vector< vector<Blob* > > unary_top_vec_;
//-------------------------------------------
	

	
};		


}  // namespace caffe

#endif 
