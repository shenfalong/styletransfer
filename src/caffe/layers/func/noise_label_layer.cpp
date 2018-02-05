
#include <vector>

#include "caffe/layers/func/noise_label_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

void NoiseLabelLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<NGPUS;i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
}


void NoiseLabelLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (bottom[0]->height() == 1 && bottom[0]->width() == 1)
	{
		if (Caffe::gan_type() == "train_dnet")
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			top[0]->Reshape(num,channels0+channels1,height,width);
			top[1]->Reshape(num,channels1,height,width);
		}
		else
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			top[0]->Reshape(2*num,channels0+channels1,height,width);
			top[1]->Reshape(2*num,channels1,height,width);
		}
	}
	else
	{
		if (Caffe::gan_type() == "train_dnet")
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			top[0]->Reshape(num,channels0,height,width);
			top[1]->Reshape(num,channels1,1,1);
			top[2]->Reshape(num,channels1,1,1);
		}
		else
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			top[0]->Reshape(2*num,channels0,height,width);
			top[1]->Reshape(2*num,channels1,1,1);
			top[2]->Reshape(2*num,channels1,1,1);
		}
	}
}

REGISTER_LAYER_CLASS(NoiseLabel);
}  // namespace caffe
