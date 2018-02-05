#ifndef CAFFE_INSTANCE_CUDNN_BATCH_NORM_LAYER_HPP_
#define CAFFE_INSTANCE_CUDNN_BATCH_NORM_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"


namespace caffe {




class InstanceCuDNNBatchNormLayer : public Layer {
 public:
  explicit InstanceCuDNNBatchNormLayer(const LayerParameter& param) : Layer(param), handles_setup_(false)  {}
  virtual ~InstanceCuDNNBatchNormLayer();
	virtual inline const char* type() const { return "InstanceCuDNNBatchNorm"; }

	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top);
  virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);

  virtual void Backward_gpu(const vector<Blob*>& top,   const vector<Blob*>& bottom);
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top);
 protected:
 	
	
	double number_collect_sample;
	bool is_initialize;
		
  bool handles_setup_;
  static cudnnHandle_t *       handle_;
  cudnnTensorDescriptor_t bottom_desc_;
  cudnnTensorDescriptor_t top_desc_;
  cudnnTensorDescriptor_t scale_bias_desc_;
  
  Blob weights;
  Blob bias;
  Blob savedmean;
  Blob savedinvvariance;  
  
  
  int num_;
  int channels_;
  int height_;
  int width_;
  
  int gpu_id_;	
};


}  // namespace caffe

#endif 
