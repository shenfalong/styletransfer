
#include <vector>

#include "caffe/layers/loss/styleloss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void StyleLossLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();



  float loss_sum;
	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, channels, channels, height*width,
														(float)1., bottom[0]->gpu_data() + bottom[0]->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(float)0., buffer_0_->mutable_gpu_data() + buffer_0_->offset(n));
		caffe_gpu_gemm(CblasNoTrans, CblasTrans, channels, channels, height*width,
														(float)1., bottom[1]->gpu_data() + bottom[1]->offset(n), bottom[1]->gpu_data() + bottom[1]->offset(n), 
														(float)0., buffer_1_->mutable_gpu_data() + buffer_1_->offset(n));					
	}
#if 0
FILE *fid = fopen("debug","wb");
fwrite(buffer_1_->cpu_data(),sizeof(float),buffer_1_->count(),fid);
fclose(fid);
LOG(FATAL)<<num<<", "<<channels<<", "<<height<<", "<<width;
#endif	
	
	caffe_gpu_add(buffer_delta_->count(), float(1),buffer_0_->gpu_data(),float(-1),buffer_1_->gpu_data(),buffer_delta_->mutable_gpu_data());
	caffe_gpu_scal(buffer_delta_->count(),float(1)/float(channels*height*width),buffer_delta_->mutable_gpu_data());		
	
	caffe_gpu_mul(buffer_square_->count(),buffer_delta_->gpu_data(),buffer_delta_->gpu_data(),buffer_square_->mutable_gpu_data());				
	

	caffe_gpu_asum(buffer_square_->count(),buffer_square_->gpu_data(),&loss_sum);											
	top[0]->mutable_cpu_data()[0] = loss_sum / float(num*channels*channels);
}


void StyleLossLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
	float loss_weight_ = top[0]->cpu_diff()[0] / float(num*channels*channels);
	caffe_copy(buffer_delta_->count(), buffer_delta_->gpu_data(), buffer_0_->mutable_gpu_diff());
	caffe_copy(buffer_delta_->count(), buffer_delta_->gpu_data(), buffer_1_->mutable_gpu_diff());
	caffe_gpu_scal(buffer_delta_->count(), float( 2)*loss_weight_ / float(channels*height*width), buffer_0_->mutable_gpu_diff());
	caffe_gpu_scal(buffer_delta_->count(), float(-2)*loss_weight_ / float(channels*height*width), buffer_1_->mutable_gpu_diff());

	for (int n = 0; n < num; n++)
	{
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels, height*width, channels,
														(float)2., buffer_0_->gpu_diff() + buffer_0_->offset(n), bottom[0]->gpu_data() + bottom[0]->offset(n), 
														(float)0., bottom[0]->mutable_gpu_diff() + bottom[0]->offset(n));		
		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels, height*width, channels,
														(float)2., buffer_1_->gpu_diff() + buffer_1_->offset(n), bottom[1]->gpu_data() + bottom[1]->offset(n), 
														(float)0., bottom[1]->mutable_gpu_diff() + bottom[1]->offset(n));		
	}	
}

void StyleLossLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
