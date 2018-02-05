#include <vector>

#include "caffe/layers/func/shortcut_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ShortcutLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{	
	caffe_gpu_add(bottom[0]->count(),float(1),bottom[0]->gpu_data(), float(1),bottom[1]->gpu_data(),top[0]->mutable_gpu_data());				
}


void ShortcutLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
	int channels = bottom[0]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();


	caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff()); 
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),bottom[1]->mutable_gpu_diff());
}

void ShortcutLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{	
	caffe_gpu_add(bottom[0]->count(),float(1),bottom[0]->gpu_sec_diff(), float(1),bottom[1]->gpu_sec_diff(),
						top[0]->mutable_gpu_sec_diff());		
}

}  // namespace caffe
