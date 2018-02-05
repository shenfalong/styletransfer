#include <vector>
#include "caffe/layers/operator/parallel_batch_norm_layer.hpp"

namespace caffe {


void ParallelBatchNormLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
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
		else 
			LOG(FATAL)<<"wrong lr_mult setting";
			
		
		this->parallel_blobs_.resize(4*NGPUS);
		
		this->parallel_blobs_[0*NGPUS] = this->blobs_[0];
		this->parallel_blobs_[1*NGPUS] = this->blobs_[1];
		this->parallel_blobs_[2*NGPUS] = this->blobs_[2];
		this->parallel_blobs_[3*NGPUS] = this->blobs_[3];			
		for (int i=1;i<NGPUS;i++)
		{
			this->parallel_blobs_[0*NGPUS+i].reset(new Blob(1,K,1,1));
			this->parallel_blobs_[1*NGPUS+i].reset(new Blob(1,K,1,1));
			this->parallel_blobs_[2*NGPUS+i].reset(new Blob(1,K,1,1));
			this->parallel_blobs_[3*NGPUS+i].reset(new Blob(1,K,1,1));
		}
  }
	parallel_mean_buffer_.resize(NGPUS);
	parallel_var_buffer_.resize(NGPUS);
  for (int i=0;i<NGPUS;i++)
	{  	
  	parallel_mean_buffer_[i]      = new Blob();
  	parallel_var_buffer_[i]  = new Blob();		    
	}
}


void ParallelBatchNormLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	CHECK_EQ(top.size(),NGPUS);
	for (int i=0;i<NGPUS;i++)
		top[i]->ReshapeLike(*bottom[i]);
		
	
	for (int i=0;i<NGPUS;i++)
	{	
		parallel_mean_buffer_[i]->Reshape(1,channels,1,1);
		parallel_var_buffer_[i]->Reshape(1,channels,1,1);
	}
}


ParallelBatchNormLayer::~ParallelBatchNormLayer() 
{
	for (int i=0;i<NGPUS;i++)
	{
		delete parallel_mean_buffer_[i];
		delete parallel_var_buffer_[i];
	}
	parallel_mean_buffer_.clear();
	parallel_var_buffer_.clear();
}


REGISTER_LAYER_CLASS(ParallelBatchNorm);
}  // namespace caffe
