#include <vector>

#include "caffe/layers/func/silence_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void SilenceLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void SilenceLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

REGISTER_LAYER_CLASS(Silence);
}  // namespace caffe
		
