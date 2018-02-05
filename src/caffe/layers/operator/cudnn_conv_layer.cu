#include <vector>


#include "caffe/layers/operator/cudnn_conv_layer.hpp"

namespace caffe {

void CuDNNConvolutionLayer::Forward_gpu( const vector<Blob*>& bottom, const vector<Blob*>& top) 
{  
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	

	CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(gpu_id_),
		    cudnn::dataType::one,
		    bottom_descs_, bottom[0]->gpu_data(),
		    filter_desc_, this->blobs_[0]->gpu_data(),
		    conv_descs_,
		    fwd_algo_, work_space0, workspace_fwd_sizes_,
		    cudnn::dataType::zero,
		    top_descs_, top[0]->mutable_gpu_data()));   		 
		            		                    
	if (this->layer_param_.convolution_param().bias_term()) 
	{
		CUDNN_CHECK(cudnnAddTensor(Caffe::cudnn_handle(gpu_id_),
		      cudnn::dataType::one,
		      bias_desc_, this->blobs_[1]->gpu_data(),
		      cudnn::dataType::one,
		      top_descs_, top[0]->mutable_gpu_data()));
	}     

}


void CuDNNConvolutionLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{  
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());
	

	if (this->layer_param_.convolution_param().bias_term() && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false) 
	{
	  CUDNN_CHECK(cudnnConvolutionBackwardBias(Caffe::cudnn_handle(gpu_id_),
	        cudnn::dataType::one,
	        top_descs_,  top[0]->gpu_diff(),
	        cudnn::dataType::one,
	        bias_desc_, this->blobs_[1]->mutable_gpu_diff()));
	}		
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(Caffe::cudnn_handle(gpu_id_),
			    cudnn::dataType::one,
			    bottom_descs_, bottom[0]->gpu_data(),
			    top_descs_,    top[0]->gpu_diff(),
			    conv_descs_,
			    bwd_filter_algo_, work_space0, workspace_bwd_filter_sizes_,
			    cudnn::dataType::one,
			    filter_desc_, this->blobs_[0]->mutable_gpu_diff()));
	}
	CUDNN_CHECK(cudnnConvolutionBackwardData(
      Caffe::cudnn_handle(gpu_id_),
      cudnn::dataType::one,
      filter_desc_, this->blobs_[0]->gpu_data(),
      top_descs_, top[0]->gpu_diff(),
      conv_descs_,
      bwd_data_algo_, work_space1, workspace_bwd_data_sizes_,
      cudnn::dataType::zero,
      bottom_descs_, bottom[0]->mutable_gpu_diff()));  
  
 #if 0 
  #if 0
  float sum = caffe_gpu_square_sum(this->blobs_[0]->count(),this->blobs_[0]->gpu_data());
  float var = float(2) /((this->kernel_size_*this->kernel_size_*(this->channels_+this->num_output_))/float(2));
  float coef = float(1e-4)*(sum - float(this->blobs_[0]->count())*var);
  //LOG(INFO)<<"weight norm = "<<sum <<" vs "<< float(this->blobs_[0]->count())*var;

  caffe_gpu_add(this->blobs_[0]->count(),float(1),this->blobs_[0]->gpu_diff(),
  																			 coef,   this->blobs_[0]->gpu_data(),
  																			 this->blobs_[0]->mutable_gpu_diff());   
 	#else
 	float sum = caffe_gpu_square_sum(this->blobs_[0]->count(),this->blobs_[0]->gpu_data());
  float var = float(2) /((this->kernel_size_*this->kernel_size_*(this->channels_+this->num_output_))/float(2));
  float coef =  float(1e-4) * (sqrt(sum) - sqrt(float(this->blobs_[0]->count())*var)) / sqrt(sum);
 		
	//LOG(INFO)<<"weight norm = "<<(sum - float(this->blobs_[0]->count())*var);
	
  caffe_gpu_add(this->blobs_[0]->count(),float(1),this->blobs_[0]->gpu_diff(),
  																			 coef,   this->blobs_[0]->gpu_data(),
  																			 this->blobs_[0]->mutable_gpu_diff());    		  	
  #endif	
 #endif 																		      							  																					      
}


void CuDNNConvolutionLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{  
	char * work_space0 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_data());
	char * work_space1 = reinterpret_cast<char *>(myworkspace_[0]->mutable_gpu_diff());

	CUDNN_CHECK(cudnnConvolutionForward(Caffe::cudnn_handle(gpu_id_),
			    cudnn::dataType::one,
			    bottom_descs_, bottom[0]->gpu_sec_diff(),
			    filter_desc_, this->blobs_[0]->gpu_data(),
			    conv_descs_,
			    fwd_algo_, work_space0, workspace_fwd_sizes_,
			    cudnn::dataType::zero,
			    top_descs_, top[0]->mutable_gpu_sec_diff()));   		         		   
  if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{     
		CUDNN_CHECK(cudnnConvolutionBackwardFilter(
			  Caffe::cudnn_handle(gpu_id_),
			  cudnn::dataType::one,
			  bottom_descs_,    bottom[0]->gpu_sec_diff(),
			  top_descs_, top[0]->gpu_diff(),
			  conv_descs_,
			  bwd_filter_algo_, work_space1, workspace_bwd_filter_sizes_,
			  cudnn::dataType::one,
			  filter_desc_, this->blobs_[0]->mutable_gpu_diff()));   
	}	

}




}  // namespace caffe
