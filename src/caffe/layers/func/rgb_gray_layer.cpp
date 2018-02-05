
#include <vector>

#include "caffe/layers/func/rgb_gray_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void RGBGRAYLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void RGBGRAYLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	CHECK_EQ(channels,3);  
  
	top[0]->Reshape(num,1,height,width);
}


REGISTER_LAYER_CLASS(RGBGRAY);
}  // namespace caffe
