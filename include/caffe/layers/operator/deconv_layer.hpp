#ifndef CAFFE_DECONV_LAYER_HPP_
#define CAFFE_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/filler.hpp"

namespace caffe {


class DeConvolutionLayer : public Layer 
{
 public:
	explicit DeConvolutionLayer(const LayerParameter& param): Layer(param) {}

  virtual inline const char* type() const { return "DeConvolution"; }
  
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);



  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
      
      
 protected:
  
  int gpu_id_;
  
	int num_output_;
  int channels_;
  int kernel_size_;
  int pad_;
  int stride_;
  int filter_stride_;
  int group_;
  
  int kernel_eff_;
	int height_out_;
	int width_out_;
	Blob * col_buffer_;
	Blob * bias_multiplier_;
};

}  // namespace caffe

#endif  // CAFFE_DECONV_LAYER_HPP_
