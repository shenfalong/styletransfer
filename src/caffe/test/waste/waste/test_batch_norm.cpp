
	#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/batch_norm_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CuDNNDeConvolutionLayerTest : public ::testing::Test
{
  typedef TypeParam Dtype;
 protected:
  CuDNNDeConvolutionLayerTest() : blob_bottom_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<48;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob<TypeParam>();
  }
  virtual void SetUp()
  {
    blob_bottom_->Reshape(2, 3, 4, 2);
   // blob_top_->Reshape(2, 3, 6, 5);

    caffe_rng_gaussian<Dtype>(this->blob_bottom_->count(), 
    													Dtype(0), Dtype(1), 
    													this->blob_bottom_->mutable_cpu_data());
    													
    //for (int i=0;i<this->blob_bottom_->count();i++)
   	//	this->blob_bottom_->mutable_cpu_data()[i]=i;	
										
    													
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~CuDNNDeConvolutionLayerTest()
  {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef testing::Types<float> myTypes;

TYPED_TEST_CASE(CuDNNDeConvolutionLayerTest, myTypes);


TYPED_TEST(CuDNNDeConvolutionLayerTest, TestGradient)
{
  typedef TypeParam Dtype;
	
  LayerParameter layer_param;
 	layer_param.mutable_batch_norm_param()->set_bn_state(LEARNED);
  BatchNormLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);

}


}  // namespace caffe
