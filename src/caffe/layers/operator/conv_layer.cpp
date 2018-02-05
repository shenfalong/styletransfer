// Copyright 2013 Yangqing Jia

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/layers/operator/conv_layer.hpp"
#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



void ConvolutionLayer::LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	CUDA_CHECK(cudaGetDevice(&gpu_id_));
	
	
  num_output_ = this->layer_param_.convolution_param().num_output();
  channels_ = bottom[0]->channels();
  kernel_size_ = this->layer_param_.convolution_param().kernel_size();
  pad_ = this->layer_param_.convolution_param().pad();
  stride_ = this->layer_param_.convolution_param().stride();
  filter_stride_ = this->layer_param_.convolution_param().filter_stride();
  group_ = this->layer_param_.convolution_param().group();
  kernel_eff_ = kernel_size_ + (kernel_size_ - 1) * (filter_stride_ - 1);
  CHECK_EQ(channels_%group_,0);
  CHECK_EQ(num_output_%group_,0);

  if (this->blobs_.size() > 0)
    LOG(INFO) << "Skipping parameter initialization";
  else
  {
    if (this->layer_param_.convolution_param().bias_term())
      this->blobs_.resize(2);
    else
      this->blobs_.resize(1);


    this->blobs_[0].reset(new Blob(num_output_, channels_/ group_, kernel_size_, kernel_size_));
    shared_ptr<Filler > weight_filler(GetFiller(this->layer_param_.convolution_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
		
		
		
	
	  if (this->layer_param_.param_size() <= 0)
	  {
	  	this->lr_mult().push_back(1);
	  	this->decay_mult().push_back(1);
	  }
		 
		
    if (this->layer_param_.convolution_param().bias_term())
    {
      this->blobs_[1].reset(new Blob(num_output_, 1, 1, 1));
      caffe_set(this->blobs_[1]->count(),float(0),this->blobs_[1]->mutable_cpu_data());
			
      if (this->layer_param_.param_size() <= 1)
			{
				this->lr_mult().push_back(2);
				this->decay_mult().push_back(0);
			}	
    }
  }
};

void ConvolutionLayer::Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
  int num = bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();



  height_out_ = (height + 2 * pad_ - kernel_eff_) / stride_ + 1;
  width_out_ = (width + 2 * pad_ - kernel_eff_) / stride_ + 1;
  
  
  top[0]->Reshape(num,num_output_,height_out_,width_out_);

//----------------------------- work space ------------------------- 
	//it seemd this operation builds a relationship between the point and the element, which I do not understand
	col_buffer_ = static_cast<Blob *>(Caffe::parallel_workspace_[gpu_id_]);
 	bias_multiplier_ = static_cast<Blob *>(Caffe::parallel_workspace_[gpu_id_+Caffe::GPUs.size()]);
//----------------------------- ------------------------------------- 
  
	col_buffer_->Reshape(kernel_size_*kernel_size_*channels,height_out_*width_out_,1,1);
  if (this->layer_param_.convolution_param().bias_term())
  {
    bias_multiplier_->Reshape(1,1,height_out_,width_out_);  
    caffe_gpu_set(bias_multiplier_->count(),float(1),bias_multiplier_->mutable_gpu_data()); 
  }

}



ConvolutionLayer::~ConvolutionLayer()
{
}



REGISTER_LAYER_CLASS(Convolution);

}  // namespace caffe
