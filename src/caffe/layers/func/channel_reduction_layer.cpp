
#include <vector>

#include "caffe/layers/func/channel_reduction_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ChannelReductionLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void ChannelReductionLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	const int num = bottom[0]->num();
	const int channels = bottom[0]->channels();
	const int height = bottom[0]->height();
	const int width = bottom[0]->width();
	CHECK_EQ(channels%2,0);			
	top[0]->Reshape(num,channels/2,height,width);
}

REGISTER_LAYER_CLASS(ChannelReduction);
}  // namespace caffe
