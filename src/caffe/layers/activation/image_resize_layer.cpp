#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>


#include "caffe/layers/activation/image_resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ImageResizeLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	interp_ratio = this->layer_param().interp_param().interp_ratio();
}


void ImageResizeLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
  int width = bottom[0]->width();
 
	top[0]->Reshape(num,channels,height/interp_ratio,width/interp_ratio);
}




REGISTER_LAYER_CLASS(ImageResize);
}  // namespace caffe
		
