
#include <vector>

#include "caffe/layers/func/trivial_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void TrivialLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
}


void TrivialLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			top[i]->ReshapeLike(*bottom[i]);
	}
}


REGISTER_LAYER_CLASS(Trivial);
}  // namespace caffe
