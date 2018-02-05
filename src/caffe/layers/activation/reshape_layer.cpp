
#include <vector>

#include "caffe/layers/activation/reshape_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ReshapeLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	num_ = this->layer_param_.shape_param().num();
	channels_ = this->layer_param_.shape_param().channels();
	height_ = this->layer_param_.shape_param().height();
	width_ = this->layer_param_.shape_param().width();
}


void ReshapeLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	if (num > 1  && this->layer_param_.shape_param().forced() == false)
	{
		//CHECK_EQ(channels_*height_*width_,channels*height*width);
		//top[0]->Reshape(num,channels_,height_,width_);
		top[0]->ReshapeLike(*bottom[0]);
	}
	else
	{
		CHECK_EQ(num_*channels_*height_*width_,num*channels*height*width);
		top[0]->Reshape(num_,channels_,height_,width_);
	}
}

REGISTER_LAYER_CLASS(Reshape);
}  // namespace caffe
