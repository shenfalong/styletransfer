
#include <vector>

#include "caffe/layers/activation/tanh_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void TanHLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void TanHLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	top[0]->ReshapeLike(*bottom[0]);
}



REGISTER_LAYER_CLASS(TanH);
}  // namespace caffe
