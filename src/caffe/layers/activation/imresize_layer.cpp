#include <vector>

#include "caffe/layers/activation/imresize_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>

namespace caffe {


void ImresizeLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	stride = this->layer_param().interp_param().stride();
	kernel_size = this->layer_param().interp_param().kernel_size();
	num_classes = this->layer_param().interp_param().num_classes();
}


void ImresizeLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  if (top.size() == 2)
	{
		top[0]->Reshape(num,num_classes,height/stride,width/stride);
		top[1]->Reshape(num,1,height/stride,width/stride);
	}
	else if (top.size() == 1)
		top[0]->Reshape(num,1,height/stride,width/stride);
}


REGISTER_LAYER_CLASS(Imresize);
}  // namespace caffe
		
