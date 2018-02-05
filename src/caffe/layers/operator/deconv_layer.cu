// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/operator/deconv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void DeConvolutionLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*> &top) 
{
  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  float* col_data = col_buffer_->mutable_gpu_data();
  const float* weight = this->blobs_[0]->gpu_data();
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
 	int bottom_offset_ = height * width * channels / group_;
 	int col_offset_ = height * width * kernel_size_ * kernel_size_ * num_output_ / group_;
 	int weight_offset_ = kernel_size_ * kernel_size_ * channels * num_output_ / group_ / group_;
 	

  for (int n = 0; n < num; n++) 
  {
  	for (int g = 0; g < group_; g++) 
  	{
		  caffe_gpu_gemm(CblasTrans, CblasNoTrans, kernel_size_*kernel_size_*num_output_/group_, height*width,  channels/group_,
														(float)1., weight + weight_offset_ * g, bottom_data + bottom[0]->offset(n) + bottom_offset_ * g,
														(float)0., col_data + col_offset_ * g);
  	}

 		
    col2im_gpu(col_data, num_output_, height_out_, width_out_,  
    kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, filter_stride_, filter_stride_, 
    top_data + top[0]->offset(n));

   
  
    if (this->layer_param_.convolution_param().bias_term()) 
    {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_output_,height_out_*width_out_, 1, 
													(float)1., this->blobs_[1]->gpu_data(), bias_multiplier_->gpu_data(),
													(float)1., top_data + top[0]->offset(n));
    }      
  }     
}


void DeConvolutionLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
//-------------------------------------------------------------------------
	if (this->layer_param_.convolution_param().bias_term())
  {
    bias_multiplier_->Reshape(1,1,height_out_,width_out_);
    caffe_gpu_set(bias_multiplier_->count(),float(1),bias_multiplier_->mutable_gpu_data());  
  }
  col_buffer_->Reshape(kernel_size_*kernel_size_*num_output_, height*width, 1, 1);
//-------------------------------------------------------------------------

  const float* top_diff = top[0]->gpu_diff();
  const float* weight = this->blobs_[0]->gpu_data();
  const float* bottom_data = bottom[0]->gpu_data();
  
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  float* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  float* col_data = col_buffer_->mutable_gpu_data();
  float* col_diff = col_buffer_->mutable_gpu_diff();




	int bottom_offset_ = height * width * channels / group_;
 	int col_offset_ = height * width * kernel_size_ * kernel_size_ * num_output_ / group_;
 	int weight_offset_ = kernel_size_ * kernel_size_ * channels * num_output_ / group_ /group_;
	

	if (this->layer_param_.convolution_param().bias_term() && this->lr_mult()[1] != 0) 
	{
		float* bias_diff = this->blobs_[1]->mutable_gpu_diff();
		for (int n = 0; n < num; ++n)  
		{
			caffe_gpu_gemv(CblasNoTrans, num_output_, height_out_ * width_out_, 
														(float)1., top_diff + top[0]->offset(n), bias_multiplier_->gpu_data(), 
														(float)1., bias_diff);
		}
	}
	
	if (this->lr_mult()[0] != 0)
	{
		for (int n = 0; n < num; ++n) 
		{
			im2col_gpu(top_diff + top[0]->offset(n), num_output_, height_out_,width_out_, 
		  kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, filter_stride_, filter_stride_, 
		  col_diff); 
		
			for (int g = 0; g < group_; ++g) 
			{
				caffe_gpu_gemm(CblasNoTrans, CblasTrans, channels/group_,  kernel_size_*kernel_size_*num_output_/group_, height*width,
															(float)1., bottom_data + bottom[0]->offset(n) + bottom_offset_ * g, col_diff + col_offset_ * g, 
															(float)1., weight_diff + weight_offset_ * g);
			}												
		}
	}
  for (int n = 0; n < num; ++n) 
  {
  	
    im2col_gpu(top_diff + top[0]->offset(n), num_output_, height_out_,width_out_, 
    kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, filter_stride_, filter_stride_, 
    col_diff);   
    
    for (int g = 0; g < group_; ++g) 
  	{
		  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, channels/group_, height*width, kernel_size_*kernel_size_*num_output_/group_,
														(float)1., weight + weight_offset_ * g, col_diff + col_offset_ * g,
														(float)0., bottom_diff + bottom[0]->offset(n) + bottom_offset_ * g);
		}												
  }    
}

void DeConvolutionLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*> &top) 
{
}


}  // namespace caffe
