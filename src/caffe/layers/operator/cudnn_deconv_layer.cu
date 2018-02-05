// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/operator/cudnn_deconv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void sync_deconv_groups() { }


void CuDNNDeConvolutionLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*> &top) 
{
	
  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  const float* weight = this->blobs_[0]->gpu_data();
  char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
 
 	CUDNN_CHECK(cudnnConvolutionBackwardData(
	      Caffe::cudnn_handle(gpu_id_),
	      cudnn::dataType::one,
	      filter_desc_, this->blobs_[0]->gpu_data(),
	      bottom_descs_, bottom_data,
	      conv_descs_,
	      bwd_data_algo_, work_space0, workspace_bwd_data_sizes_,
	      cudnn::dataType::zero,
	      top_descs_, top_data)); 	
	if (this->layer_param_.convolution_param().bias_term()) 
	{
	  const float* bias_data = this->blobs_[1]->gpu_data();
	  CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(gpu_id_),
	        cudnn::dataType::one,
	        bias_desc_, bias_data,
	        cudnn::dataType::one,
	        top_descs_, top_data));
	}           
}


void CuDNNDeConvolutionLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
  const float* top_diff = top[0]->gpu_diff();
  const float* top_data = top[0]->gpu_data();
  const float* weight = this->blobs_[0]->gpu_data();
  
  const float* bottom_data = bottom[0]->gpu_data();
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  
  char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
		

	if (this->layer_param_.convolution_param().bias_term() && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false) 
	{
	  CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(gpu_id_),
	        cudnn::dataType::one,
	        top_descs_,  top_diff,
	        cudnn::dataType::one,
	        bias_desc_, this->blobs_[1]->mutable_gpu_diff()));
	}
	CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(gpu_id_),
	      cudnn::dataType::one,
	      top_descs_, top_diff,
	      filter_desc_, weight,
	      conv_descs_,
	      fwd_algo_, work_space0, workspace_fwd_sizes_,
	      cudnn::dataType::zero,
	      bottom_descs_, bottom_diff));   
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{          
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(
	      Caffe::cudnn_handle(gpu_id_),
	      cudnn::dataType::one,
	      top_descs_,    top_diff,
	      bottom_descs_, bottom_data,
	      conv_descs_,
	      bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
	      cudnn::dataType::one,
	      filter_desc_, this->blobs_[0]->mutable_gpu_diff()));           
	}    
}

void CuDNNDeConvolutionLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*> &top) 
{
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	CUDNN_CHECK(cudnnConvolutionBackwardData(
		      Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType::one,
		      filter_desc_, this->blobs_[0]->gpu_data(),
		      bottom_descs_, bottom[0]->gpu_sec_diff(),
		      conv_descs_,
		      bwd_data_algo_, work_space0, workspace_bwd_data_sizes_,
		      cudnn::dataType::zero,
		      top_descs_, top[0]->mutable_gpu_sec_diff())); 		   

  if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{     
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(
			  Caffe::cudnn_handle(gpu_id_),
			  cudnn::dataType::one,
			  top_descs_,    top[0]->gpu_diff(),
			  bottom_descs_, bottom[0]->gpu_sec_diff(),
			  conv_descs_,
			  bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
			  cudnn::dataType::one,
			  filter_desc_, this->blobs_[0]->mutable_gpu_diff()));   
	}

}


}  // namespace caffe
