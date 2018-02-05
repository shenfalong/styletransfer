
#include <vector>

#include "caffe/layers/loss/euclideanloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void EuclideanLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	buffer_delta_ = new Blob();
	buffer_square_ = new Blob();
}


void EuclideanLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	
	buffer_delta_->Reshape(num,channels,height,width);
	buffer_square_->Reshape(num,channels,height,width);
	top[0]->Reshape(1,1,1,1);
}



REGISTER_LAYER_CLASS(EuclideanLoss);
}  // namespace caffe
