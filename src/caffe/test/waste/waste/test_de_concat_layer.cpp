
	#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/de_concat_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class DeConcatLayerTest : public ::testing::Test
{
  typedef TypeParam Dtype;
 protected:
  DeConcatLayerTest() : blob_bottom_(new Blob<Dtype>()), blob_top_0_(new Blob<Dtype>()), blob_top_1_(new Blob<Dtype>())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<12;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob<TypeParam>();
  }
  virtual void SetUp()
  {
    blob_bottom_->Reshape(2, 4, 2, 3);

    caffe_rng_gaussian<Dtype>(this->blob_bottom_->count(), 
    													Dtype(0), Dtype(1), 
    													this->blob_bottom_->mutable_cpu_data());
    													
    													
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_0_);
    blob_top_vec_.push_back(blob_top_1_);
  }
  virtual ~DeConcatLayerTest()
  {
    delete blob_bottom_;
    delete blob_top_0_;
    delete blob_top_1_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_0_;
  Blob<Dtype>* const blob_top_1_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef testing::Types<float,double> myTypes;

TYPED_TEST_CASE(DeConcatLayerTest, myTypes);


TYPED_TEST(DeConcatLayerTest, TestGradient)
{
  typedef TypeParam Dtype;

  LayerParameter layer_param;
  layer_param.mutable_concat_param()->add_channels(3);
  layer_param.mutable_concat_param()->add_channels(1);
  DeConcatLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);

}


}  // namespace caffe
