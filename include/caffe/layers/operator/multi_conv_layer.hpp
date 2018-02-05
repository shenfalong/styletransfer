#ifndef CAFFE_MULTI_CONV_LAYER_HPP_
#define CAFFE_MULTI_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"

namespace caffe {


class MultiConvolutionLayer : public Layer 
{
 public:
  explicit MultiConvolutionLayer(const LayerParameter& param): Layer(param) {}
	virtual ~MultiConvolutionLayer();
  virtual inline const char* type() const { return "MultiConvolution"; }
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);



  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
      
      
 protected:
  
	int num_output_;
  int channels_;
  int kernel_size_;
  int pad_;
  int stride_;
  int filter_stride_;
  int group_;
  int multi_;
  int multi_num_output_;
  
  int kernel_eff_;
	int height_out_;
	int width_out_;

//------------------------------------------	
	Blob * col_buffer_;
	Blob * bias_multiplier_;
	Blob * buffer_top_;
	int gpu_id_;
//------------------------------------------	
};

}  // namespace caffe

#endif  // CAFFE_CONV_LAYER_HPP_
