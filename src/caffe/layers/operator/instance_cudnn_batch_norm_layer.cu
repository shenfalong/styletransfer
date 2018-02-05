
#include <vector>

#include "caffe/layers/operator/instance_cudnn_batch_norm_layer.hpp"

namespace caffe {



void InstanceCuDNNBatchNormLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for (int n=0;n < bottom[0]->num();n++)
	{
		CUDNN_CHECK(cudnnBatchNormalizationForwardTraining(Caffe::cudnn_handle(gpu_id_),
			    CUDNN_BATCHNORM_SPATIAL,
			    cudnn::dataType::one,cudnn::dataType::zero,
			    bottom_desc_, bottom[0]->gpu_data() + bottom[0]->offset(n),
			    top_desc_, top[0]->mutable_gpu_data() + top[0]->offset(n),
			    scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[1]->gpu_data(),
			    float(1),
			    this->blobs_[2]->mutable_gpu_data(),this->blobs_[3]->mutable_gpu_data(),
			    double(CUDNN_BN_MIN_EPSILON),
			    savedmean.mutable_gpu_data()+savedmean.offset(n),savedinvvariance.mutable_gpu_data()+savedinvvariance.offset(n)));		        
	}   
}


void InstanceCuDNNBatchNormLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	
	const float* top_data = top[0]->gpu_data();
	const float* top_diff = top[0]->gpu_diff();
	const float* bottom_data = bottom[0]->gpu_data();
	float* bottom_diff = bottom[0]->mutable_gpu_diff();
	for (int n=0;n < bottom[0]->num();n++)
	{
		CUDNN_CHECK(cudnnBatchNormalizationBackward(Caffe::cudnn_handle(gpu_id_),
					CUDNN_BATCHNORM_SPATIAL,
					cudnn::dataType::one,cudnn::dataType::zero,
					cudnn::dataType::one,cudnn::dataType::one,
					bottom_desc_, bottom[0]->gpu_data() + bottom[0]->offset(n),
					top_desc_,top[0]->gpu_diff() + top[0]->offset(n),
					bottom_desc_, bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n),
					scale_bias_desc_,this->blobs_[0]->gpu_data(),this->blobs_[0]->mutable_gpu_diff(),this->blobs_[1]->mutable_gpu_diff(),
					double(CUDNN_BN_MIN_EPSILON),
					savedmean.gpu_data()+savedmean.offset(n),savedinvvariance.gpu_data()+savedinvvariance.offset(n)));		
	}
}

void InstanceCuDNNBatchNormLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
