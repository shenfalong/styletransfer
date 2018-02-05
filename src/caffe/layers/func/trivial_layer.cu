
#include <vector>

#include "caffe/layers/func/trivial_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void TrivialLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			caffe_copy(bottom[i]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
	}
}


void TrivialLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			caffe_copy(top[i]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
	}
}


void TrivialLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for (int i=0;i<bottom.size();i++)
	{
		if (i<top.size())
			caffe_copy(bottom[i]->count(),bottom[0]->gpu_sec_diff(),top[0]->mutable_gpu_sec_diff());
	}
}


}  // namespace caffe
