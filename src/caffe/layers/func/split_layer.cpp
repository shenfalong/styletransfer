#include <vector>

#include "caffe/layers/func/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void SplitLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
}


void SplitLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
  CHECK_GE(top.size(),1);
	
  for (int i = 0; i < top.size(); ++i)
    top[i]->ReshapeLike(*bottom[0]);
}



REGISTER_LAYER_CLASS(Split);

}  // namespace caffe
