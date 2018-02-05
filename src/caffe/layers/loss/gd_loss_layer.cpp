
#include <vector>

#include "caffe/layers/loss/gd_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void GdLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void GdLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	CHECK_EQ(bottom.size(),1);
	CHECK_EQ(num%2,0);
	CHECK_EQ(channels,1);

	loss_.Reshape(num/2,1,height,width);
	top[0]->Reshape(1,1,1,1);
}



REGISTER_LAYER_CLASS(GdLoss);
}  // namespace caffe
