#ifndef CAFFE_CUDNN_CONV_LAYER_HPP_
#define CAFFE_CUDNN_CONV_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/operator/conv_layer.hpp"

namespace caffe {



class CuDNNConvolutionLayer : public ConvolutionLayer 
{
 public:
  explicit CuDNNConvolutionLayer(const LayerParameter& param) : ConvolutionLayer(param), handles_setup_(false) {}
  virtual ~CuDNNConvolutionLayer();
	virtual inline const char* type() const { return "CuDNNConvolution"; }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
  

	
  bool handles_setup_;
  int gpu_id_;

	
  // algorithms for forward and backwards convolutions
  cudnnConvolutionFwdAlgo_t fwd_algo_;
  cudnnConvolutionBwdFilterAlgo_t bwd_filter_algo_;
  cudnnConvolutionBwdDataAlgo_t bwd_data_algo_;

  cudnnTensorDescriptor_t bottom_descs_, top_descs_;
  cudnnTensorDescriptor_t    bias_desc_;
  cudnnFilterDescriptor_t      filter_desc_;
  cudnnConvolutionDescriptor_t conv_descs_;
  int bottom_offset_, top_offset_, bias_offset_;

  size_t workspace_fwd_sizes_;
  size_t workspace_bwd_data_sizes_;
  size_t workspace_bwd_filter_sizes_;

	vector<Blob *> myworkspace_;
	
	int iter_;
	Blob v_blob_;
	Blob m_blob_;
	Blob buffer_blob_;
	
};


}  // namespace caffe

#endif  // CAFFE_CUDNN_CONV_LAYER_HPP_
