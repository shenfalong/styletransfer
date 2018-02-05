
#include <vector>

#include "caffe/layers/func/concat_num_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ConcatNumLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	
}


void ConcatNumLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
 	int num0 = bottom[0]->num();
 	int num1 = bottom[1]->num();
 	int num2 = 0;
 	if (bottom.size() == 3)
 		num2 = bottom[2]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(),bottom[1]->height());
  CHECK_EQ(bottom[0]->width(),bottom[1]->width());
  if (bottom.size() == 3)
  {
  	CHECK_EQ(bottom[0]->channels(),bottom[2]->channels());
  	CHECK_EQ(bottom[0]->height(),bottom[2]->height());
  	CHECK_EQ(bottom[0]->width(),bottom[2]->width());
  }
  top[0]->Reshape(num0+num1+num2,channels,height,width);
}



REGISTER_LAYER_CLASS(ConcatNum);
}  // namespace caffe
