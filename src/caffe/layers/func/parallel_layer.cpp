#include <vector>

#include "caffe/layers/func/parallel_layer.hpp"

namespace caffe {

void ParallelLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	
	LayerParameter layer_param(this->layer_param_);
  
  //LOG(INFO)<<"building layer "<<layer_param.type();
  
  unary_bottom_vec_.resize(NGPUS);
  unary_top_vec_.resize(NGPUS);
  unary_layer_.resize(NGPUS);
  
  CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	unary_layer_[0] = LayerRegistry::CreateLayer(layer_param);
	
	
	unary_bottom_vec_[0].clear();
	for (int k=0;k<bottom.size();k += NGPUS)
		unary_bottom_vec_[0].push_back(bottom[k]);
	unary_top_vec_[0].clear();
	for (int k=0;k<top.size();k += NGPUS)
		unary_top_vec_[0].push_back(top[k]);	
	unary_layer_[0]->SetUp(unary_bottom_vec_[0], unary_top_vec_[0]);
 
  
  int num_blobs=unary_layer_[0]->blobs().size();
  
  this->parallel_blobs_.resize(num_blobs*NGPUS);
  for (int j=0;j<num_blobs;j++)
			this->parallel_blobs_[j*NGPUS] = unary_layer_[0]->blobs()[j];	
  for(int i=1;i<NGPUS;i++)
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i] = LayerRegistry::CreateLayer(layer_param);
		unary_bottom_vec_[i].clear();
		for (int k=0;k<bottom.size();k += NGPUS)
			unary_bottom_vec_[i].push_back(bottom[k+i]);
		unary_top_vec_[i].clear();
		for (int k=0;k<top.size();k += NGPUS)
			unary_top_vec_[i].push_back(top[k+i]);	
		unary_layer_[i]->SetUp(unary_bottom_vec_[i], unary_top_vec_[i]);
		
		for (int j=0;j<num_blobs;j++)
			this->parallel_blobs_[j*NGPUS+i] = unary_layer_[i]->blobs()[j];	
	}
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	
	
	this->blobs_.resize(num_blobs);
	for (int i=0;i<num_blobs;i++)
		this->blobs_[i] = this->parallel_blobs_[i*NGPUS];

	this->lr_mult() = unary_layer_[0]->lr_mult();
	this->decay_mult() = unary_layer_[0]->decay_mult();
}


void ParallelLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for(int i=0;i<NGPUS;i++)
	{
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));
		unary_layer_[i]->Reshape(unary_bottom_vec_[i],unary_top_vec_[i]);
	}
	CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
	
	//lambda layer has modified layer_param_;
	this->layer_param_ = unary_layer_[0]->layer_param();
	if (this->layer_param_.type() == "Lambda")
		LOG(FATAL)<<"lambda layer should be instanced !!";
}

ParallelLayer::~ParallelLayer() 
{
}

REGISTER_LAYER_CLASS(Parallel);
}  // namespace caffe

