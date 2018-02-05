// Copyright 2013 Yangqing Jia
#ifndef CAFFE_LAYER_H_
#define CAFFE_LAYER_H_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/layer_factory.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/math_functions.hpp"

using std::vector;

namespace caffe {

class Layer
{
 public:
  explicit Layer(const LayerParameter& param): layer_param_(param) 
  {
  	lr_mult_.clear();
		decay_mult_.clear();
  	for(int i=0;i<param.param_size();i++)
  	{
  		lr_mult_.push_back(param.param(i).lr_mult());
			decay_mult_.push_back(param.param(i).decay_mult());
  	}
		has_bottom_sec_diff_ = false;
  }
  virtual ~Layer(){};

  
  void SetUp(const vector<Blob*>& bottom, const vector<Blob*>& top)  
  {     
  	LayerSetUp(bottom, top); 
  	Reshape(bottom, top); 
  }
  
  inline float Forward(const vector<Blob*>& bottom, const vector<Blob*>& top);
  inline void SecForward(const vector<Blob*>& bottom, const vector<Blob*>& top);
  inline void Backward(const vector<Blob*>& top, const vector<Blob*>& bottom);


  vector<shared_ptr<Blob> >& blobs() { return blobs_; }
  vector<shared_ptr<Blob> >& first_moment() { return first_moment_; }
  vector<shared_ptr<Blob> >& second_moment() { return second_moment_; }
  vector<shared_ptr<Blob> >& parallel_blobs() { return parallel_blobs_; }
	vector<float>& lr_mult() { return lr_mult_; }
	vector<float>& decay_mult() { return decay_mult_; }
	

  const LayerParameter& layer_param() { return layer_param_; }
  virtual void ToProto(LayerParameter* param, bool write_diff = false);
	virtual void compute_sec_loss(const vector<Blob*>& top, const float sec_loss_weight, const float norm_value);
	
	virtual void LayerSetUp(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;
	virtual void Reshape(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;
  virtual void Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;
  virtual void SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) = 0;
  virtual void Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) = 0;
	
 protected:
  LayerParameter layer_param_;
  vector<float> lr_mult_;
	vector<float> decay_mult_;
  vector<shared_ptr<Blob > > blobs_;
  vector<shared_ptr<Blob > > first_moment_;
  vector<shared_ptr<Blob > > second_moment_;
  
  vector<shared_ptr<Blob > > parallel_blobs_;
	bool has_bottom_sec_diff_;
	

	
  DISABLE_COPY_AND_ASSIGN(Layer);
};  

inline float Layer::Forward(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	//LOG(INFO)<<"-----------processing "<<this->layer_param_.type()<<", top.size() = "<<top.size();

	Forward_gpu(bottom, top);

	float loss_weight = layer_param_.include().loss_weight();
	
	float loss = 0;
	if (loss_weight > 0)
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());
		for (int i=0;i<top.size();i++)
			loss += top[i]->cpu_data()[0] * loss_weight / float(top.size());
	}

	return loss;
};



inline void Layer::Backward(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	float loss_weight = layer_param_.include().loss_weight();
	if (loss_weight > 0)
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());	
		for (int i=0;i<top.size();i++)
			top[i]->mutable_cpu_diff()[0] = loss_weight / float(top.size());
	}
	
  Backward_gpu(top, bottom);	
};


inline void Layer::SecForward(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	//LOG(INFO)<<"-----------processing "<<this->layer_param_.type()<<", top.size() = "<<top.size();
	float sec_loss_weight = layer_param_.include().sec_loss_weight();
	float norm_value = layer_param_.include().norm_value();
	
	if (sec_loss_weight > 0 && Caffe::second_pass())
	{
		CHECK_EQ(Caffe::GPUs.size(),top.size());	
		compute_sec_loss(top,sec_loss_weight / float(top.size()), norm_value);
	}
	else
	{
		SecForward_gpu(bottom, top);
	}
};

//---------------------------------------------------------------------------------------
}  // namespace caffe

#endif  // CAFFE_LAYER_H_
