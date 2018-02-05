
#include <vector>

#include "caffe/layers/func/lambda_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void LambdaLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
	all_layer_.resize(this->layer_param_.branch_size());
	unary_bottom_vec_.resize(this->layer_param_.branch_size());
	for (int i=0;i<this->layer_param_.branch_size();i++)
	{
		all_layer_[i] = LayerRegistry::CreateLayer(this->layer_param_.branch(i));
		
		unary_bottom_vec_[i].clear();
		for (int j=0;j<this->layer_param_.branch(i).bottom_index_size();j++)
		{
			int k = this->layer_param_.branch(i).bottom_index(j);
			CHECK_LE(k,bottom.size()-1);
			unary_bottom_vec_[i].push_back(bottom[k]);
		}
		all_layer_[i]->LayerSetUp(unary_bottom_vec_[i], top);
	}
	
	int all_index = 0;
	for (int layer_index_=0;layer_index_<all_layer_.size();layer_index_++)
	{
		for (int i_index=0;i_index<all_layer_[layer_index_]->blobs().size();i_index++)
		{
			this->lr_mult()[all_index] = all_layer_[layer_index_]->lr_mult()[i_index];
			this->decay_mult()[all_index] = all_layer_[layer_index_]->decay_mult()[i_index];
			this->blobs()[all_index] = all_layer_[layer_index_]->blobs()[i_index];
			all_index++;
		}
	}
}


void LambdaLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	layer_index_ = -1;
	for (int i=0;i<this->layer_param_.branch_size();i++)
	{
		if (Caffe::compare_variable(this->layer_param_.bind_key(), this->layer_param_.branch(i).bind_value()))
		{
			layer_index_ = i;
			break;
		}
	}	
	if (layer_index_ == -1)
		LOG(FATAL)<<this->layer_param_.bind_key()<<", "<<this->layer_param_.branch(0).bind_value();
	

	this->layer_param_.mutable_include()->set_loss_weight(this->layer_param_.branch(layer_index_).include().loss_weight());
	this->layer_param_.mutable_include()->set_sec_loss_weight(this->layer_param_.branch(layer_index_).include().sec_loss_weight());
	this->layer_param_.set_type(this->layer_param_.branch(layer_index_).type());
	

	
	all_layer_[layer_index_]->Reshape(unary_bottom_vec_[layer_index_], top); 
}


REGISTER_LAYER_CLASS(Lambda);
}  // namespace caffe
