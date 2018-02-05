#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/loss/softmax_loss_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class SoftmaxWithLossLayerTest : public ::testing::Test
{
  typedef TypeParam Dtype;
 protected:
  SoftmaxWithLossLayerTest() : blob_bottom_0_(new Blob<Dtype>()),blob_bottom_1_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<12;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob<TypeParam>();
  }
  virtual void SetUp()
  {
    blob_bottom_0_->Reshape(2, 5, 3, 4);
    blob_bottom_1_->Reshape(2, 1, 3, 4);

    caffe_rng_gaussian<Dtype>(this->blob_bottom_0_->count(), Dtype(2), Dtype(1), 
    													this->blob_bottom_0_->mutable_cpu_data());			
    													
										
    for (int i=0;i<blob_bottom_1_->count();i++)
    	blob_bottom_1_->mutable_cpu_data()[i] = caffe_rng_rand()%5;
    
    													
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~SoftmaxWithLossLayerTest()
  {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef testing::Types<float> myTypes;

TYPED_TEST_CASE(SoftmaxWithLossLayerTest, myTypes);


TYPED_TEST(SoftmaxWithLossLayerTest, TestGradient)
{
  typedef TypeParam Dtype;

  LayerParameter layer_param;
  layer_param.mutable_loss_param()->set_ignore_label(255);
  SoftmaxWithLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  //checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_,0);
	checker.CheckSecGradientExhaustive(&layer,  this->blob_top_vec_, this->blob_bottom_vec_,0);
}


}  // namespace caffe
