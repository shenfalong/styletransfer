#include "caffe/layers/loss/gan_smooth_l1_loss_layer.hpp"

namespace caffe {


void GANSmoothL1LossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	ignore_value = this->layer_param_.loss_param().ignore_label();
}


void GANSmoothL1LossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	top[0]->Reshape(1,1,1,1);
	
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  
  loss_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
  counts_.Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width());
}



REGISTER_LAYER_CLASS(GANSmoothL1Loss);

}  // namespace caffe
