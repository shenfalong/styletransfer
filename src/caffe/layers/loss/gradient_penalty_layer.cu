
#include <vector>

#include "caffe/layers/loss/gradient_penalty_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void GradientPenaltyLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();

	CHECK_EQ(bottom.size(),1);
	//CHECK_EQ(channels,1);
	
	top[0]->mutable_cpu_data()[0] = caffe_gpu_sum(bottom[0]->count(),bottom[0]->gpu_data());	
}


void GradientPenaltyLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (Caffe::second_pass() == false)
	{
		int num = bottom[0]->num();
		int channels = bottom[0]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
		
		float loss_weights_ = top[0]->cpu_diff()[0];
		caffe_gpu_set(bottom[0]->count(),loss_weights_,bottom[0]->mutable_gpu_diff());	
	}
	else
	{
		caffe_gpu_set(bottom[0]->count(),float(0),bottom[0]->mutable_gpu_diff());
	}
}

void GradientPenaltyLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
