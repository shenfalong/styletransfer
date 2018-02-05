
#include <vector>

#include "caffe/layers/operator/mean_variance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void MeanVarianceLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void MeanVarianceLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	top[0]->Reshape(num,2*channels,1,1);
}




REGISTER_LAYER_CLASS(MeanVariance);
}  // namespace caffe
