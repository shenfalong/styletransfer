
#include <vector>

#include "caffe/layers/activation/scale_bias_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ScaleBiasLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

	classes_ = this->layer_param_.noise_param().classes();
	
	#if 0
	if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {

    this->blobs_.resize(2);
	
  	int channels = bottom[0]->channels();  
    this->blobs_[0].reset(new Blob(classes_,channels,1,1));
   	this->blobs_[1].reset(new Blob(classes_,channels,1,1));
		caffe_set(this->blobs_[0]->count(),float(1),this->blobs_[0]->mutable_cpu_data());
    caffe_set(this->blobs_[1]->count(),float(0),this->blobs_[1]->mutable_cpu_data());
		
    if (this->lr_mult().size() == 0)
    {
    	this->lr_mult().push_back(1);
    	this->decay_mult().push_back(1);
    }	
  }
  #endif
}


void ScaleBiasLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}


REGISTER_LAYER_CLASS(ScaleBias);
}  // namespace caffe
