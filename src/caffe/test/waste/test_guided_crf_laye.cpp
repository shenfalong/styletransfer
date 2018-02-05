#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/operator/guided_crf_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class TemplateLayerTest : public ::testing::Test
{
  typedef TypeParam Dtype;
protected:
  TemplateLayerTest() : blob_bottom_(new Blob<Dtype>()), blob_bottom_1_(new Blob<Dtype>()), blob_top_(new Blob<Dtype>())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<12;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob<TypeParam>();
  }
  virtual void SetUp()
  {
    blob_bottom_->Reshape(2, 4, 6, 8);
    blob_bottom_1_->Reshape(2, 3, 6, 8);

    caffe_rng_gaussian<Dtype>(this->blob_bottom_->count(),
                              Dtype(0), Dtype(1),
                              this->blob_bottom_->mutable_cpu_data());

    for (int i=0;i<blob_bottom_1_->count();i++)
      blob_bottom_1_->mutable_cpu_data()[i]=Dtype(caffe_rng_rand()% 255);

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~TemplateLayerTest()
  {
    delete blob_bottom_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_bottom_1_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef testing::Types<float,double> myTypes;

TYPED_TEST_CASE(TemplateLayerTest, myTypes);


TYPED_TEST(TemplateLayerTest, TestGradient)
{
  typedef TypeParam Dtype;

  LayerParameter layer_param;
  layer_param.mutable_crf_param()->set_max_iter(1);
  layer_param.mutable_crf_param()->set_radius(Dtype(3));
  layer_param.mutable_crf_param()->set_eps(Dtype(1));
  layer_param.mutable_crf_param()->set_alpha(Dtype(-2));
  layer_param.add_param()->set_lr_mult(Dtype(0));
  
  GuidedCRFLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_,0);

}


}  // namespace caffe
