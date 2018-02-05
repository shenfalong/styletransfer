#include <vector>

#include "caffe/layers/activation/box_filter_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void BoxFilterLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  
  box_filter_gpu(num,channels,height,width,radius,bottom[0]->gpu_data(),top[0]->mutable_gpu_data(),buffer_.mutable_gpu_data());

}


void BoxFilterLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	box_filter_gpu(num,channels,height,width,radius,top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff(),buffer_.mutable_gpu_data());
}


void BoxFilterLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

}

}  // namespace caffe
		
