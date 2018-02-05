
#include <vector>

#include "caffe/layers/func/parallel_layer.hpp"
#include <omp.h>
namespace caffe {


void ParallelLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
//-------------------------------------------------------
	if (NGPUS > 1)
	{
		if ((this->layer_param_.type() == "CuDNNBatchNorm"))
		{	
			if (Caffe::number_collect_sample != -1)
			{
				CHECK_EQ(this->parallel_blobs_.size(),4*NGPUS);	
				for (int i = 0; i < NGPUS; i++) 
				{  	
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclBcast((void *)this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[2*NGPUS+i]->count(),
			 																		ncclFloat,0,Caffe::comms(i),NULL);			
				}		
				for (int i = 0; i < NGPUS; i++) 
				{  	
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclBcast((void *)this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[3*NGPUS+i]->count(),
			 																		ncclFloat,0,Caffe::comms(i),NULL);			
				}		
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
			}	 
		}
		if (this->layer_param_.type() == "BatchNorm")
		{
			if (Caffe::number_collect_sample != -1)
			{
				CHECK_EQ(this->parallel_blobs_.size(),2*NGPUS);
				for (int i = 0; i < NGPUS; i++) 
				{  	
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclBcast((void *)this->parallel_blobs_[0*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[0*NGPUS+i]->count(),
			 																		ncclFloat,0,Caffe::comms(i),NULL);			
				}		
				for (int i = 0; i < NGPUS; i++) 
				{  	
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclBcast((void *)this->parallel_blobs_[1*NGPUS+i]->mutable_gpu_data(),this->parallel_blobs_[1*NGPUS+i]->count(),
			 																		ncclFloat,0,Caffe::comms(i),NULL);			
				}		
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
			}	 
		}
	}
//-------------------------------------------------------
omp_set_num_threads(NGPUS);
#pragma omp parallel for
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->Forward_gpu(unary_bottom_vec_[i], unary_top_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
//-------------------------------------------------------
	if (NGPUS > 1)
	{
		if ((this->layer_param_.type() == "CuDNNBatchNorm"))
		{
			if (Caffe::number_collect_sample != -1)
			{
			
				for(int i=0;i<NGPUS;i++)
				{ 
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclReduce( this->parallel_blobs_[2*NGPUS+i]->gpu_data(),this->parallel_blobs_[2*NGPUS+i]->mutable_gpu_data(),
							this->parallel_blobs_[2*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
				}
			
				for(int i=0;i<NGPUS;i++)
				{ 
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclReduce( this->parallel_blobs_[3*NGPUS+i]->gpu_data(),this->parallel_blobs_[3*NGPUS+i]->mutable_gpu_data(),
							this->parallel_blobs_[3*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
				}
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
				caffe_gpu_scal(this->blobs_[2]->count(),float(1)/float(NGPUS),this->blobs_[2]->mutable_gpu_data());
				caffe_gpu_scal(this->blobs_[3]->count(),float(1)/float(NGPUS),this->blobs_[3]->mutable_gpu_data());	
			
			}	
		}
		if ((this->layer_param_.type() == "BatchNorm"))
		{
			if (Caffe::number_collect_sample != -1)
			{
			
				for(int i=0;i<NGPUS;i++)
				{ 
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclReduce( this->parallel_blobs_[0*NGPUS+i]->gpu_data(),this->parallel_blobs_[0*NGPUS+i]->mutable_gpu_data(),
							this->parallel_blobs_[0*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
				}
			
				for(int i=0;i<NGPUS;i++)
				{ 
					CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
					ncclReduce( this->parallel_blobs_[1*NGPUS+i]->gpu_data(),this->parallel_blobs_[1*NGPUS+i]->mutable_gpu_data(),
							this->parallel_blobs_[1*NGPUS+i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
				}
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
				caffe_gpu_scal(this->blobs_[0]->count(),float(1)/float(NGPUS),this->blobs_[0]->mutable_gpu_data());
				caffe_gpu_scal(this->blobs_[1]->count(),float(1)/float(NGPUS),this->blobs_[1]->mutable_gpu_data());	
			
			}	
		}
		if (this->layer_param_.type() == "BeGdLoss")
		{
		
			for(int i=0;i<NGPUS;i++)
			{ 
				CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
				ncclReduce(this->parallel_blobs_[i]->gpu_data(),this->parallel_blobs_[i]->mutable_gpu_data(),
						this->parallel_blobs_[i]->count(), ncclFloat,ncclSum,0,Caffe::comms(i),NULL);
			}
			CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
			caffe_gpu_scal(this->blobs_[0]->count(),float(1)/float(NGPUS),this->blobs_[0]->mutable_gpu_data());
		
		}
	}
//-------------------------------------------------------
}


void ParallelLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	omp_set_num_threads(NGPUS);
	#pragma omp parallel for
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->Backward_gpu(unary_top_vec_[i],unary_bottom_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}

void ParallelLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	omp_set_num_threads(NGPUS);
	#pragma omp parallel for
	for (int i = 0; i < NGPUS; ++i) 
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->SecForward_gpu(unary_bottom_vec_[i], unary_top_vec_[i]);
	}  
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
}

}  // namespace caffe

