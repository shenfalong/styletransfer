
	#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/style_per_loss_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class StylePerLossLayerTest : public ::testing::Test
{
  typedef TypeParam Dtype;
 protected:
  StylePerLossLayerTest() : blob_bottom_0_(new Blob<Dtype>()),blob_bottom_1_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<12;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob<TypeParam>();
  }
  virtual void SetUp()
  {
    blob_bottom_0_->Reshape(4, 3, 2, 3);
    blob_bottom_1_->Reshape(4, 3, 2, 3);
		
#if 0		
		blob_bottom_0_->mutable_cpu_data()[0] = 0;
		blob_bottom_0_->mutable_cpu_data()[1] = 1;
		blob_bottom_0_->mutable_cpu_data()[2] = 2;
		
		blob_bottom_1_->mutable_cpu_data()[0] = 0;
		blob_bottom_1_->mutable_cpu_data()[1] = 0;
		blob_bottom_1_->mutable_cpu_data()[2] = 0;
#endif		
    caffe_rng_gaussian<Dtype>(this->blob_bottom_0_->count(), 
    													Dtype(0), Dtype(1), 
    													this->blob_bottom_0_->mutable_cpu_data());
    caffe_rng_gaussian<Dtype>(this->blob_bottom_1_->count(), 
    													Dtype(0), Dtype(1), 
    													this->blob_bottom_1_->mutable_cpu_data());													
    													
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~StylePerLossLayerTest()
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

typedef testing::Types<float,double> myTypes;

TYPED_TEST_CASE(StylePerLossLayerTest, myTypes);


TYPED_TEST(StylePerLossLayerTest, TestGradient)
{
  typedef TypeParam Dtype;

  LayerParameter layer_param;
  StylePerLossLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-8);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_,0);

}


}  // namespace caffe
