
#include <vector>

#include "caffe/layers/loss/gradient_penalty_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void GradientPenaltyLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void GradientPenaltyLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	top[0]->Reshape(1,1,1,1);
}



REGISTER_LAYER_CLASS(GradientPenalty);
}  // namespace caffe
