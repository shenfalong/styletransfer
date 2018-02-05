#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/parallel_batch_norm_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class CuDNNDeConvolutionLayerTest : public ::testing::Test
{
  typedef TypeParam Dtype;
 protected:
  CuDNNDeConvolutionLayerTest() : blob_bottom_0_(new Blob<Dtype>()),blob_bottom_1_(new Blob<Dtype>()), blob_top_0_(new Blob<Dtype>()), blob_top_1_(new Blob<Dtype>())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<48;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob<TypeParam>();
  }
  virtual void SetUp()
  {
    blob_bottom_0_->Reshape(2, 3, 2, 3);
		blob_bottom_1_->Reshape(2, 3, 2, 3);

    caffe_rng_gaussian<Dtype>(this->blob_bottom_0_->count(), 
    													Dtype(0), Dtype(1), 
    													this->blob_bottom_0_->mutable_cpu_data());
    													
		caffe_rng_gaussian<Dtype>(this->blob_bottom_1_->count(), 
    													Dtype(0), Dtype(1), 
    													this->blob_bottom_1_->mutable_cpu_data());
		this->blob_bottom_0_->mutable_cpu_data()[0] = 0;
		this->blob_bottom_0_->mutable_cpu_data()[1] = 1;
		
		this->blob_bottom_1_->mutable_cpu_data()[0] = 2;
		this->blob_bottom_1_->mutable_cpu_data()[1] = 3;
		
										
    													
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_0_);
    blob_top_vec_.push_back(blob_top_1_);
  }
  virtual ~CuDNNDeConvolutionLayerTest()
  {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_0_;
    delete blob_top_1_;
  }
  Blob<Dtype>* const blob_bottom_0_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef testing::Types<float> myTypes;

TYPED_TEST_CASE(CuDNNDeConvolutionLayerTest, myTypes);


TYPED_TEST(CuDNNDeConvolutionLayerTest, TestGradient)
{
  typedef TypeParam Dtype;

  LayerParameter layer_param;
  ParallelBatchNormLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);

}


}  // namespace caffe
