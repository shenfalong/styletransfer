#include <vector>

#include "caffe/layers/operator/cudnn_batch_norm_layer.hpp"

namespace caffe {


void CuDNNBatchNormLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
	mean_buffer_ = new Blob();
	var_buffer_ = new Blob();
	
  cudnn::createTensor4dDesc(&bottom_desc_);
  cudnn::createTensor4dDesc(&top_desc_);
  cudnn::createTensor4dDesc(&scale_bias_desc_);
  
  
  if (this->blobs_.size() == 4)
  	LOG(INFO)<<"skip initialization";
  else 
  {
    const int K = bottom[0]->channels();
    this->blobs_.resize(4);
    for(int i=0;i<this->blobs_.size();i++)
    {
      this->blobs_[i].reset(new Blob());
      this->blobs_[i]->Reshape(1,K,1,1);
    }
    float std = 0.02;
    //caffe_rng_gaussian(this->blobs_[0]->count(), float(1), std, this->blobs_[0]->mutable_cpu_data());
    //caffe_rng_gaussian(this->blobs_[1]->count(), float(0), std, this->blobs_[1]->mutable_cpu_data());
    caffe_set(this->blobs_[0]->count(),float(1),this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(),float(0),this->blobs_[1]->mutable_cpu_data());
    caffe_set(this->blobs_[2]->count(),float(0),this->blobs_[2]->mutable_cpu_data());
    caffe_set(this->blobs_[3]->count(),float(1),this->blobs_[3]->mutable_cpu_data());
		

		if (this->layer_param_.param_size() == 2)
	  { 
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
	  }	
		else if(this->layer_param_.param_size() == 0)
		{		
			this->lr_mult().push_back(1);
		  this->decay_mult().push_back(1);
		  this->lr_mult().push_back(1);
		  this->decay_mult().push_back(1);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		  this->lr_mult().push_back(0);
		  this->decay_mult().push_back(0);
		} 
  }

  
  is_initialize = false;
}


void CuDNNBatchNormLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();



  top[0]->Reshape(num,channels,height,width);    

	mean_buffer_->Reshape(1,channels,1,1);
  var_buffer_->Reshape(1,channels,1,1);

  cudnn::setTensor4dDesc(&bottom_desc_, num, channels, height, width);
  cudnn::setTensor4dDesc(&top_desc_, num, channels, height, width);
  cudnn::setTensor4dDesc(&scale_bias_desc_,  1, channels, 1, 1);

}


CuDNNBatchNormLayer::~CuDNNBatchNormLayer() 
{
	cudnnDestroyTensorDescriptor(bottom_desc_);
	cudnnDestroyTensorDescriptor(top_desc_);
	cudnnDestroyTensorDescriptor(scale_bias_desc_);
	
	delete mean_buffer_;
	delete var_buffer_;
}


REGISTER_LAYER_CLASS(CuDNNBatchNorm);
}  // namespace caffe
