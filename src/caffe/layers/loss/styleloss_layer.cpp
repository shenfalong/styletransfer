
#include <vector>

#include "caffe/layers/loss/styleloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void StyleLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	buffer_0_ = new Blob();
	buffer_1_ = new Blob();
	buffer_delta_ = new Blob();
	buffer_square_ = new Blob();
}


void StyleLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	buffer_0_->Reshape(num,1,channels,channels);
	buffer_1_->Reshape(num,1,channels,channels);
	buffer_delta_->Reshape(num,1,channels,channels);
	buffer_square_->Reshape(num,1,channels,channels);
	top[0]->Reshape(1,1,1,1);
}



REGISTER_LAYER_CLASS(StyleLoss);
}  // namespace caffe
