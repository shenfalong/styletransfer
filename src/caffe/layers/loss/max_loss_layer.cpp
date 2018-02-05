
#include <vector>

#include "caffe/layers/loss/max_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void MaxLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void MaxLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CHECK_EQ(bottom.size(),2);
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	loss_.Reshape(num,1,height,width);
	
	top[0]->Reshape(1,1,1,1);
}


REGISTER_LAYER_CLASS(MaxLoss);
}  // namespace caffe
