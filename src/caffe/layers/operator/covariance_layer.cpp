
#include <vector>

#include "caffe/layers/operator/covariance_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void CovarianceLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void CovarianceLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	top[0]->Reshape(num,channels*channels,1,1);
}


REGISTER_LAYER_CLASS(Covariance);
}  // namespace caffe
