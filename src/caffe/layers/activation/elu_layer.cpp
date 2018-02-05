#include <algorithm>
#include <vector>

#include "caffe/layers/activation/elu_layer.hpp"

namespace caffe {


void ELULayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

}



void ELULayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
  top[0]->ReshapeLike(*bottom[0]);
}


REGISTER_LAYER_CLASS(ELU);

}  // namespace caffe

