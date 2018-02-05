
#include <vector>

#include "caffe/layers/func/separate_blob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void SeparateBlobLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{

}


void SeparateBlobLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  


	top[0]->Reshape(num/2, channels, height, width);
	top[1]->Reshape(num/2, channels, height, width);
}



REGISTER_LAYER_CLASS(SeparateBlob);
}  // namespace caffe
