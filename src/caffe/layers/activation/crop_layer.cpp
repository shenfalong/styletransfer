#include <vector>

#include "caffe/layers/activation/crop_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void CropLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	pad_ = this->layer_param_.convolution_param().pad();
}


void CropLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
	top[0]->Reshape(num,channels,height-pad_*2,width-pad_*2);
}


REGISTER_LAYER_CLASS(Crop);
}  // namespace caffe
		
