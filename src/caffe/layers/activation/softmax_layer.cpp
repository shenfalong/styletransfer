#include <algorithm>
#include <vector>

#include "caffe/layers/activation/softmax_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void SoftmaxLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int channels = bottom[0]->channels();
	
	
	sum_multiplier_.Reshape(1, channels, 1, 1);
	caffe_set(sum_multiplier_.count(), float(1), sum_multiplier_.mutable_cpu_data());
}



void SoftmaxLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();

  top[0]->ReshapeLike(*bottom[0]);
  
	
  scale_.Reshape(num,1,height,width); 
}



REGISTER_LAYER_CLASS(Softmax);

}  // namespace caffe
