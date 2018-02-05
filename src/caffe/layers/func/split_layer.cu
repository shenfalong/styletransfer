#include <vector>

#include "caffe/layers/func/split_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void SplitLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

	for (int i = 1; i < top.size(); ++i)
		caffe_copy(top[i]->count(),bottom[0]->gpu_data(),top[i]->mutable_gpu_data());

}


void SplitLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
  
  caffe_gpu_add(bottom[0]->count(), float(1), top[0]->gpu_diff(), float(1),top[1]->gpu_diff(),  
  										bottom[0]->mutable_gpu_diff());
  for (int i = 2; i < top.size(); ++i) 
  {
    const float* top_diff = top[i]->gpu_diff();
    float* bottom_diff = bottom[0]->mutable_gpu_diff();
    caffe_gpu_axpy(bottom[0]->count(), float(1.), top_diff, bottom_diff);
  }
}

void SplitLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	for (int i = 1; i < top.size(); ++i)
			caffe_copy(top[i]->count(),bottom[0]->gpu_sec_diff(),top[i]->mutable_gpu_sec_diff());
}



}  // namespace caffe]);
