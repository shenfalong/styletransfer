
#include <vector>

#include "caffe/layers/loss/gan_multi_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void GANMultiSoftmaxWithLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  CHECK_EQ(channels%2,0);
  CHECK_EQ(height,1);
  CHECK_EQ(width,1);
  bottom[0]->Reshape(num*channels/2,2,1,1);
  CHECK_EQ(bottom[1]->channels(),channels/2);
  bottom[1]->Reshape(num*channels/2,1,1,1);
  
	LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);
}


void GANMultiSoftmaxWithLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  CHECK_EQ(channels%2,0);
  CHECK_EQ(height,1);
  CHECK_EQ(width,1);
  bottom[0]->Reshape(num*channels/2,2,1,1);
  bottom[1]->Reshape(num*channels/2,1,1,1);
  
	softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  top[0]->Reshape(1,1,1,1);
  if (Caffe::gan_type() == "train_dnet")
  	loss_.Reshape(bottom[0]->num()/2,1,bottom[0]->height(),bottom[0]->width());
  else
  	loss_.Reshape(bottom[0]->num(),1,bottom[0]->height(),bottom[0]->width());
}


REGISTER_LAYER_CLASS(GANMultiSoftmaxWithLoss);
}  // namespace caffe
