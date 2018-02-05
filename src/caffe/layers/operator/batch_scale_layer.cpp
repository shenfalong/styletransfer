
#include <vector>

#include "caffe/layers/operator/batch_scale_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void BatchScaleLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
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
  }
}


void BatchScaleLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	sum_.Reshape(1,channels,1,1);
	top[0]->ReshapeLike(*bottom[0]);
}


REGISTER_LAYER_CLASS(BatchScale);
}  // namespace caffe
