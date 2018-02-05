
#include <vector>

#include "caffe/layers/func/pad_image_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void PadImageLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	pad_ = this->layer_param_.convolution_param().pad();
}


void PadImageLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
  
	top[0]->Reshape(num,channels,height+2*pad_,width+2*pad_);
}


REGISTER_LAYER_CLASS(PadImage);
}  // namespace caffe
