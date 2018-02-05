#include <algorithm>
#include <vector>

#include "caffe/layers/operator/cudnn_conv_layer.hpp"

namespace caffe {

#define CUDNN_STREAMS 3



void CuDNNConvolutionLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
  ConvolutionLayer::LayerSetUp(bottom, top);
	
	iter_ = 0;
//----------------------------------------	
	myworkspace_.resize(1);
	myworkspace_[0] = static_cast<Blob *>(Caffe::parallel_workspace_[gpu_id_]);
//----------------------------------------	


  cudnn::createFilterDesc(&filter_desc_,
      this->num_output_, this->channels_/this->group_, this->kernel_size_, this->kernel_size_);
  if (this->layer_param_.convolution_param().bias_term()) 
  {
    cudnn::createTensor4dDesc(&bias_desc_);
   	cudnn::setTensor4dDesc(&bias_desc_, 1, this->num_output_, 1, 1); 
  } 

  cudnn::createTensor4dDesc(&bottom_descs_);
  cudnn::createTensor4dDesc(&top_descs_);
  cudnn::createConvolutionDesc(&conv_descs_);      
}


void CuDNNConvolutionLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  ConvolutionLayer::Reshape(bottom, top);

  const int num = bottom[0]->num();
  const int channels = bottom[0]->channels();
  const int height = bottom[0]->height();
  const int width = bottom[0]->width();
  const int height_out = top[0]->height();
  const int width_out = top[0]->width();

	cudnn::setTensor4dDesc(&bottom_descs_,
		    num, this->channels_, height, width);
	cudnn::setTensor4dDesc(&top_descs_,
		    num, this->num_output_, height_out, width_out);  

	cudnn::setConvolutionDesc(&conv_descs_, 
		    this->pad_, this->pad_, this->stride_, this->stride_, this->filter_stride_, this->filter_stride_, this->group_);
  //set the max work space data in case of RUNOUT of memory
  //take 448 x 448 as a exemplar
	size_t workspace_limit_bytes = 1011888000;
	if (num == 1)
		workspace_limit_bytes = 0;

  	CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(Caffe::cudnn_handle(gpu_id_),
					bottom_descs_,
					filter_desc_,
					conv_descs_,
					top_descs_,
					CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
					workspace_limit_bytes,
					&fwd_algo_));			
		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(Caffe::cudnn_handle(gpu_id_),
		    bottom_descs_, 
		    top_descs_, 
		    conv_descs_, 
		    filter_desc_,
		    CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
		    workspace_limit_bytes, 
		    &bwd_filter_algo_) );
		CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(Caffe::cudnn_handle(gpu_id_),
		    filter_desc_, 
		    top_descs_, 
		    conv_descs_, 
		    bottom_descs_,
		    CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
		    workspace_limit_bytes, 
		    &bwd_data_algo_));   

            
	//fwd_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	//bwd_filter_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
	//bwd_data_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;    


   
	CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
			bottom_descs_,
			filter_desc_,
			conv_descs_,
			top_descs_,
			fwd_algo_,
			&(workspace_fwd_sizes_)));			   
	CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
		  bottom_descs_, 
		  top_descs_, 
		  conv_descs_, 
		  filter_desc_,
		  bwd_filter_algo_, 
		  &workspace_bwd_filter_sizes_));    
	CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(Caffe::cudnn_handle(gpu_id_),
			filter_desc_, 
			top_descs_, 
			conv_descs_, 
			bottom_descs_,
			bwd_data_algo_, 
			&workspace_bwd_data_sizes_) );   

  //LOG(INFO)<<" fwd_algo_ = "<<fwd_algo_ <<" "
 // 					<<" bwd_filter_algo_ ="<<bwd_filter_algo_<<" "
  //					<<" bwd_data_algo_ = "<<bwd_data_algo_;    
  
//-----------------------------------------------------------------------------------------	
	myworkspace_[0]->Reshape(workspace_fwd_sizes_/sizeof(float)+1,1,1,1);
 	myworkspace_[0]->Reshape(workspace_bwd_data_sizes_/sizeof(float)+1,1,1,1);
 	myworkspace_[0]->Reshape(workspace_bwd_filter_sizes_/sizeof(float)+1,1,1,1);   
 	myworkspace_[0]->gpu_data(); 
 	myworkspace_[0]->gpu_diff(); 
//-----------------------------------------------------------------------------------------	   
       
}

CuDNNConvolutionLayer::~CuDNNConvolutionLayer() 
{

  CUDNN_CHECK(cudnnDestroyTensorDescriptor(bottom_descs_));
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(top_descs_));
  CUDNN_CHECK(cudnnDestroyConvolutionDescriptor(conv_descs_));
  
  if (this->layer_param_.convolution_param().bias_term()) 
    CUDNN_CHECK(cudnnDestroyTensorDescriptor(bias_desc_));

  CUDNN_CHECK(cudnnDestroyFilterDescriptor(filter_desc_));
}


REGISTER_LAYER_CLASS(CuDNNConvolution);
}   // namespace caffe

