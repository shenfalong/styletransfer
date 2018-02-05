// TODO (sergeyk): effect should not be dependent on phase. wasted memcpy.

#include <vector>

#include "caffe/layers/activation/dropout_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void DropoutLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  threshold_ = this->layer_param_.dropout_param().dropout_ratio();
  DCHECK(threshold_ > 0.);
  DCHECK(threshold_ < 1.);
}


void DropoutLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();



	top[0]->ReshapeLike(*bottom[0]);
	
  rand_vec_.Reshape(1,channels,1,1);
}



REGISTER_LAYER_CLASS(Dropout);

}  // namespace caffe
