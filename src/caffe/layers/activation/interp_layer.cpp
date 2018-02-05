#include <vector>

#include "caffe/layers/activation/interp_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void InterpLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	interp_ratio = this->layer_param_.interp_param().interp_ratio();
}


void InterpLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (bottom.size() == 2)
  	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),bottom[1]->height(),bottom[1]->width());
  else
  	top[0]->Reshape(bottom[0]->num(),bottom[0]->channels(),interp_ratio*bottom[0]->height(),interp_ratio*bottom[0]->width());
}



REGISTER_LAYER_CLASS(Interp);
}  // namespace caffe
