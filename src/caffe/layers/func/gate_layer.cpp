
#include <vector>

#include "caffe/layers/func/gate_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void GateLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<NGPUS;i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	

}


void GateLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	
	
	top[0]->Reshape(num,channels,height,width);
}

REGISTER_LAYER_CLASS(Gate);
}  // namespace caffe
