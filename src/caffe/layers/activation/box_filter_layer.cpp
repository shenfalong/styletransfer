#include <vector>

#include "caffe/layers/activation/box_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void BoxFilterLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	radius =this->layer_param_.crf_param().radius();
}


void BoxFilterLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	buffer_.ReshapeLike(*bottom[0]);
	top[0]->ReshapeLike(*bottom[0]);
}


REGISTER_LAYER_CLASS(BoxFilter);
}  // namespace caffe
		
