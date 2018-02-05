#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/operator/conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void ConvolutionLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*> &top) 
{
  const float* bottom_data = bottom[0]->gpu_data();
  float* top_data = top[0]->mutable_gpu_data();
  float* col_data = col_buffer_->mutable_gpu_data();
  const float* weight = this->blobs_[0]->gpu_data();
  
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
 
 	int top_offset_ = height_out_ * width_out_ * num_output_ / group_;
 	int col_offset_ = height_out_ * width_out_ * kernel_size_ * kernel_size_ * channels / group_;
 	int weight_offset_ = kernel_size_ * kernel_size_ * channels * num_output_ / group_ / group_;

  for (int n = 0; n < num; ++n) 
  {
    im2col_gpu(bottom_data + bottom[0]->offset(n), channels, height,width, 
    kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, filter_stride_, filter_stride_, 
    col_data);   
    
    for (int g = 0; g < group_; g++) 
  	{
		  caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_output_/ group_, height_out_*width_out_, kernel_size_*kernel_size_*channels/ group_,
														(float)1., weight+ weight_offset_ * g , col_data + col_offset_ * g,
														(float)0., top_data + top[0]->offset(n) + top_offset_ * g );  
		}												
    if (this->layer_param_.convolution_param().bias_term()) 
    {
      caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, num_output_,height_out_*width_out_, 1, 
														(float)1., this->blobs_[1]->gpu_data(), bias_multiplier_->gpu_data(),
														(float)1., top_data + top[0]->offset(n));
    }    
  }     
}


void ConvolutionLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{	
	int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
//---------------------------------------------------------------------------------------------------
	col_buffer_->Reshape(kernel_size_*kernel_size_*channels,height_out_*width_out_,1,1);
	if (this->layer_param_.convolution_param().bias_term())
  {
    bias_multiplier_->Reshape(1,1,height_out_,width_out_);   
    caffe_gpu_set(bias_multiplier_->count(),float(1),bias_multiplier_->mutable_gpu_data()); 
  }
//---------------------------------------------------------------------------------------------------

  const float* top_diff = top[0]->gpu_diff();
  const float* weight = this->blobs_[0]->gpu_data();
  const float* bottom_data = bottom[0]->gpu_data();
  
  float* bottom_diff = bottom[0]->mutable_gpu_diff();
  float* weight_diff = this->blobs_[0]->mutable_gpu_diff();
  float* col_data = col_buffer_->mutable_gpu_data();
  float* col_diff = col_buffer_->mutable_gpu_diff();


  if (this->layer_param_.convolution_param().bias_term() && this->lr_mult()[1] > 0 && Caffe::frozen_param() == false) 
  {
		float* bias_diff = this->blobs_[1]->mutable_gpu_diff();
		for (int n = 0; n < num; ++n)  
		{
			caffe_gpu_gemv(CblasNoTrans, num_output_, height_out_ * width_out_, 
		  			(float)1., top_diff + top[0]->offset(n), bias_multiplier_->gpu_data(), 
		    		(float)1., bias_diff);
		}
  }
	
	int top_offset_ = height_out_ * width_out_ * num_output_ / group_;
 	int col_offset_ = height_out_ * width_out_ * kernel_size_ * kernel_size_ * channels / group_;
 	int weight_offset_ = kernel_size_ * kernel_size_ * channels * num_output_ / group_ / group_;
 	
	if (this->lr_mult()[0] > 0 && Caffe::frozen_param() == false)
	{
		for (int n = 0; n < num; ++n) 
		{
			im2col_gpu(bottom_data + bottom[0]->offset(n), channels, height,width, 
		  kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, filter_stride_, filter_stride_, 
		  col_data);   
		
			for (int g = 0; g < group_; g++) 
			{
				caffe_gpu_gemm(CblasNoTrans, CblasTrans, num_output_ / group_, kernel_size_*kernel_size_*channels / group_, height_out_*width_out_,
															(float)1., top_diff + top[0]->offset(n) + top_offset_ * g, col_data + col_offset_ * g, 
															(float)1., weight_diff + weight_offset_ * g);
			}												
		}
	}
	
  for (int n = 0; n < num; ++n) 
  {
  	for (int g = 0; g < group_; g++) 
  	{
		  caffe_gpu_gemm(CblasTrans, CblasNoTrans, kernel_size_*kernel_size_*channels/ group_, height_out_*width_out_, num_output_/ group_,
														(float)1., weight + weight_offset_ * g, top_diff + top[0]->offset(n) + top_offset_ * g,
														(float)0., col_diff + col_offset_ * g);
  	}
    col2im_gpu(col_diff,  channels, height, width,  
    kernel_size_, kernel_size_, pad_, pad_, stride_, stride_, filter_stride_, filter_stride_, 
    bottom_diff + bottom[0]->offset(n));
  }     
}


void ConvolutionLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*> &top) 
{

}



}  // namespace caffe
