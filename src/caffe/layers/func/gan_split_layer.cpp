
#include <vector>

#include "caffe/layers/func/gan_split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void GANSplitLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<NGPUS;i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
}


void GANSplitLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CHECK_GE(top.size(),1);
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  top[0]->Reshape(num,channels,height,width);
 	if (Caffe::gan_type() == "train_dnet")
  	top[1]->Reshape(num,channels,height,width);
  else
  	top[1]->Reshape(num*2,channels,height,width);
}

REGISTER_LAYER_CLASS(GANSplit);
}  // namespace caffe
