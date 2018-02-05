#include <algorithm>
#include <vector>

#include "caffe/layers/activation/relu_layer.hpp"

namespace caffe {


void ReLULayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
	negative_slope = this->layer_param_.relu_param().negative_slope();
}



void ReLULayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	
	
  top[0]->ReshapeLike(*bottom[0]);
  flag.Reshape((bottom[0]->count()+3)/4,1,1,1);
}

REGISTER_LAYER_CLASS(ReLU);
}  // namespace caffe
