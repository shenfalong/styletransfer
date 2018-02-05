
#include <vector>

#include "caffe/layers/activation/one_hot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void OneHotLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	classes_ = this->layer_param_.noise_param().classes();
}


void OneHotLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	top[0]->Reshape(num,classes_,height,width);
}


REGISTER_LAYER_CLASS(OneHot);
}  // namespace caffe
