
	#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/func/gate_layer.hpp"

#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

class GateLayerTest : public ::testing::Test
{
 protected:
  GateLayerTest() : blob_bottom_0_(new Blob()),blob_bottom_1_(new Blob()), blob_top_(new Blob())
  {
  	caffe::Caffe::parallel_workspace_.resize(48);
		for (int i=0;i<12;i++)
			caffe::Caffe::parallel_workspace_[i] = new caffe::Blob();
  }
  virtual void SetUp()
  {
    blob_bottom_0_->Reshape(2, 3, 6, 5);
    blob_bottom_1_->Reshape(2, 3, 6, 5);

    caffe_rng_gaussian(this->blob_bottom_0_->count(), float(0), float(1), this->blob_bottom_0_->mutable_cpu_data());
    caffe_rng_gaussian(this->blob_bottom_1_->count(), float(0), float(1), this->blob_bottom_1_->mutable_cpu_data());													
    													
    blob_bottom_vec_.push_back(blob_bottom_0_);
    blob_bottom_vec_.push_back(blob_bottom_1_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GateLayerTest()
  {
    delete blob_bottom_0_;
    delete blob_bottom_1_;
    delete blob_top_;
  }
  Blob* const blob_bottom_0_;
  Blob* const blob_bottom_1_;
  Blob* const blob_top_;
  vector<Blob*> blob_bottom_vec_;
  vector<Blob*> blob_top_vec_;
};


TEST_F(GateLayerTest, TestGradient)
{
  LayerParameter layer_param;
  GateLayer layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Reshape(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  
  GradientChecker checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_, this->blob_top_vec_);
	//checker.CheckSecGradientExhaustive(&layer,  this->blob_top_vec_, this->blob_bottom_vec_);
}


}  // namespace caffe
