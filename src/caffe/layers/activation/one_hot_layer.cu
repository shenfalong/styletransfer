
#include <vector>

#include "caffe/layers/activation/one_hot_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void kernel(int count, int channels,int spatial_dim, const float *in, float *out)
{

	CUDA_KERNEL_LOOP(i, count)
	{
		int n = i / spatial_dim;
		int s = i % spatial_dim;
		
		int label_index = in[i];
		out[(n*channels+label_index)*spatial_dim+s] = 1;
	}
}


void OneHotLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
	
	caffe_gpu_set(top[0]->count(),float(0),top[0]->mutable_gpu_data());
	kernel<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>
	(bottom[0]->count(),classes_,height*width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
}


void OneHotLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	
}


void OneHotLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
