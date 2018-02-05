#ifndef CAFFE_CUDNN_DECONV_LAYER_HPP_
#define CAFFE_CUDNN_DECONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/operator/deconv_layer.hpp"

namespace caffe {




class CuDNNDeConvolutionLayer : public DeConvolutionLayer 
{
 public:
  explicit CuDNNDeConvolutionLayer(const LayerParameter& param) : DeConvolutionLayer(param) {}
  virtual ~CuDNNDeConvolutionLayer();
	virtual inline const char* type() const { return "CuDNNDeConvolution"; }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
  
 protected:
  
	
	
  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

	size_t workspace_fwd_sizes_;
  size_t workspace_bwd_data_sizes_;
  size_t workspace_bwd_filter_sizes_;

  cudnnTensorDescriptor_t bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  cudnnConvolutionDescriptor_t conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;
	
	
	vector<Blob *> myworkspace_;
	
	int gpu_id_;
};


}  // namespace caffe

#endif  // CAFFE_CUDNN_DECONV_LAYER_HPP_
