
#include <vector>

#include "caffe/layers/loss/tv_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void TVLossLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
}


void TVLossLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	loss_.ReshapeLike(*bottom[0]);
	top[0]->Reshape(1,1,1,1);
}



REGISTER_LAYER_CLASS(TVLoss);
}  // namespace caffe
