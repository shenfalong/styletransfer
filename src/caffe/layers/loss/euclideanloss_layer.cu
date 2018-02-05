
#include <vector>

#include "caffe/layers/loss/euclideanloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void EuclideanLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	caffe_gpu_add(buffer_delta_->count(),float(1),bottom[0]->gpu_data(),float(-1),bottom[1]->gpu_data(),buffer_delta_->mutable_gpu_data());
	caffe_gpu_mul(buffer_square_->count(),buffer_delta_->gpu_data(),buffer_delta_->gpu_data(),buffer_square_->mutable_gpu_data());
	caffe_gpu_scal(buffer_square_->count(),float(1)/float(channels*height*width),buffer_square_->mutable_gpu_data());					
	
	float loss_sum;
	caffe_gpu_asum(buffer_square_->count(),buffer_square_->gpu_data(),&loss_sum);
	
	top[0]->mutable_cpu_data()[0] = loss_sum / num;			
}


void EuclideanLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	float loss_weight_ = top[0]->cpu_diff()[0] / num;
	caffe_copy(bottom[0]->count(), buffer_delta_->gpu_data(), bottom[0]->mutable_gpu_diff());
	caffe_copy(bottom[1]->count(), buffer_delta_->gpu_data(), bottom[1]->mutable_gpu_diff());
	caffe_gpu_scal(bottom[0]->count(), float( 2)*loss_weight_ / float(channels*height*width), bottom[0]->mutable_gpu_diff());
	caffe_gpu_scal(bottom[1]->count(), float(-2)*loss_weight_ / float(channels*height*width), bottom[1]->mutable_gpu_diff());
}

void EuclideanLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
