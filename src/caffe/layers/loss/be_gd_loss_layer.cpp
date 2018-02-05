
#include <vector>

#include "caffe/layers/loss/be_gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void BeGdLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	
	if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {
   
    this->blobs_.resize(1);
    this->blobs_[0].reset(new Blob(1,1,1,1));
    
    
		this->blobs_[0]->mutable_cpu_data()[0] = 0.00001;//in case of zero gradients

	  if (this->layer_param_.param_size() <= 0)
	  {
	  	this->lr_mult().push_back(1);
	  	this->decay_mult().push_back(1);
	  }
  }
}


void BeGdLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
	CHECK_EQ(bottom.size(),2);
	CHECK_EQ(num%2,0);
	
	loss_d_.Reshape(num/2,channels,height,width);
	loss_g_.Reshape(num/2,channels,height,width);
	top[0]->Reshape(1,1,1,1);
}



REGISTER_LAYER_CLASS(BeGdLoss);
}  // namespace caffe
