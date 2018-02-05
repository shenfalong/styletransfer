#include <cmath>
#include <vector>

#include "caffe/layers/activation/sigmoid_layer.hpp"

namespace caffe {



void SigmoidLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}



void SigmoidLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
  top[0]->ReshapeLike(*bottom[0]);
}




REGISTER_LAYER_CLASS(Sigmoid);

}  // namespace caffe
