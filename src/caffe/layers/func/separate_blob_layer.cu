
#include <vector>

#include "caffe/layers/func/separate_blob_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void SeparateBlobLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
#if 0
FILE *fid = fopen("debug","wb");
fwrite(bottom[0]->cpu_data(),sizeof(float),bottom[0]->count(),fid);
fclose(fid);
LOG(FATAL)<<num<<", "<<channels<<", "<<height<<", "<<width;
#endif
  
	
	caffe_copy(top[0]->count(),bottom[0]->gpu_data()                         ,top[0]->mutable_gpu_data());
	caffe_copy(top[1]->count(),bottom[0]->gpu_data()+bottom[0]->offset(num/2),top[1]->mutable_gpu_data());
}


void SeparateBlobLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  caffe_gpu_set(bottom[0]->count(),float(0),bottom[0]->mutable_gpu_diff());
  
	
	caffe_copy(top[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
	caffe_copy(top[1]->count(),top[1]->gpu_diff(),bottom[0]->mutable_gpu_diff()+bottom[0]->offset(num/2));

}

void SeparateBlobLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	caffe_copy(top[0]->count(),bottom[0]->gpu_sec_diff()                         ,top[0]->mutable_gpu_sec_diff());
	caffe_copy(top[1]->count(),bottom[0]->gpu_sec_diff()+bottom[0]->offset(num/2),top[1]->mutable_gpu_sec_diff());
}

}  // namespace caffe
