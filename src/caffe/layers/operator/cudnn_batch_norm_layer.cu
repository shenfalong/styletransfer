#include <vector>
#include "caffe/layers/operator/cudnn_batch_norm_layer.hpp"
#define BN_EPS float(1e-5)


namespace caffe {

static __global__ void linear_batch_norm_forward(int num,int channels,int height,int width,
													const float *weight,const float * in, const float * bias, float *out)
{
  CUDA_KERNEL_LOOP(ind,num*channels*height*width)
  {
  	int c = ind / width / height % channels;
  	out[ind] = weight[c] * in[ind] + bias[c];
  }
}


static __global__ void linear_batch_norm_backward(int num,int channels,int height,int width,
													const float *weight,const float * in, const float * bias, float *out)
{
  CUDA_KERNEL_LOOP(ind,num*channels*height*width)
  {
  	int c = ind / width / height % channels;
  	out[ind] = weight[c] * in[ind];
  }
}


void CuDNNBatchNormLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (Caffe::bn_state() == "frozen")
	{
		const int K = bottom[0]->channels();
		weights.Reshape(1,K,1,1);
		bias.Reshape(1,K,1,1);
	
		for(int c=0;c<K;c++)
		{
			weights.mutable_cpu_data()[c] = this->blobs_[0]->cpu_data()[c] / (sqrtf(this->blobs_[3]->cpu_data()[c]+ float(CUDNN_BN_MIN_EPSILON)));
			bias.mutable_cpu_data()[c] = -this->blobs_[0]->cpu_data()[c]*this->blobs_[2]->cpu_data()[c] / (sqrtf(this->blobs_[3]->cpu_data()[c] + float(CUDNN_BN_MIN_EPSILON)))
																								+this->blobs_[1]->cpu_data()[c];															
		}				
	} 	

	if (Caffe::number_collect_sample == 0 && Caffe::bn_state() == "learned")
	{
		caffe_gpu_set(this->blobs_[2]->count(),float(0),this->blobs_[2]->mutable_gpu_data());
		caffe_gpu_set(this->blobs_[3]->count(),float(0),this->blobs_[3]->mutable_gpu_data());
	}

  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  
   
	if (Caffe::bn_state() == "learned")
	{	
		double factor;
		if (Caffe::number_collect_sample == -1)
			factor = 0.01;
		else 
			factor = double(1)/double(Caffe::number_collect_sample+1);
	 

		CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(Caffe::cudnn_handle(gpu_id_),
		      CUDNN_BATCHNORM_SPATIAL,
		      cudnn::dataType::one,cudnn::dataType::zero,
		      bottom_desc_, bottom_data,
		      top_desc_,top_data,
		      scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
		      factor,
		      this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
		      double(CUDNN_BN_MIN_EPSILON),
		      mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	     
		      
  }  
	else
	{

		linear_batch_norm_forward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width(),
		weights.gpu_data(),bottom[0]->gpu_data(),bias.gpu_data(),top[0]->mutable_gpu_data()); 
/*
		CUDNN_CHECK(cudnnBatchNormalizationForwardInference(Caffe::cudnn_handle(gpu_id_),
		      CUDNN_BATCHNORM_SPATIAL,
		      cudnn::dataType::one,cudnn::dataType::zero,
		      bottom_desc_, bottom_data,
		      top_desc_,top_data,
		      scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
		      this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
		      double(0.001)
		      ));	       	       
*/         
	}	   	        
	
	
}


void CuDNNBatchNormLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::bn_state() == "learned")
  {
		const float* top_data = top[0]->gpu_data();
		const float* top_diff = top[0]->gpu_diff();
		const float* bottom_data = bottom[0]->gpu_data();
		float* bottom_diff = bottom[0]->mutable_gpu_diff();
		if (Caffe::frozen_param() == false)
		{		
			CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType::one,cudnn::dataType::zero,
						cudnn::dataType::one,cudnn::dataType::one,
						bottom_desc_, bottom_data,
						top_desc_,top_diff,
						bottom_desc_, bottom_diff,
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_diff(),
						double(CUDNN_BN_MIN_EPSILON),
						mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	   						
		}
		else
		{		
			CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType::one,cudnn::dataType::zero,
						cudnn::dataType::zero,cudnn::dataType::one,
						bottom_desc_, bottom_data,
						top_desc_,top_diff,
						bottom_desc_, bottom_diff,
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_diff(),//not use
						double(CUDNN_BN_MIN_EPSILON),
						mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	   	
		}
  }    
  else
  {
  	linear_batch_norm_backward<<<CAFFE_GET_BLOCKS(top[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
		(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width(),
		weights.gpu_data(),top[0]->gpu_diff(),bias.gpu_data(),bottom[0]->mutable_gpu_diff());  
  } 
}

void CuDNNBatchNormLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
						CUDNN_BATCHNORM_SPATIAL,
						cudnn::dataType::one,cudnn::dataType::zero,
						cudnn::dataType::one,cudnn::dataType::one,
						bottom_desc_, bottom[0]->gpu_data(),
						bottom_desc_,bottom[0]->gpu_sec_diff(),
						top_desc_, top[0]->mutable_gpu_sec_diff(),
						scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_sec_diff(),//blobs_[1]->diff shoud be fixed
						double(CUDNN_BN_MIN_EPSILON),
						mean_buffer_->mutable_gpu_data(),var_buffer_->mutable_gpu_data()));	   
}


}  // namespace caffe
