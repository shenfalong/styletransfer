// Fillers are random number generators that fills a blob using the specified
// algorithm. The expectation is that they are only going to be used during
// initialization time and will not involve any GPUs.

#ifndef CAFFE_FILLER_HPP
#define CAFFE_FILLER_HPP

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

/// @brief Fills a Blob with constant or randomly-generated data.

class Filler 
{
 public:
  explicit Filler(const FillerParameter& param) : filler_param_(param) 
  {}
  virtual ~Filler() 
  {}
  virtual void Fill(Blob* blob) = 0;
 protected:
  FillerParameter filler_param_;
};  // class Filler


class BinaryFiller : public Filler 
{
 public:
  explicit BinaryFiller(const FillerParameter& param) : Filler(param) 
  {}
  virtual void Fill(Blob* blob) 
  {
    CHECK(blob->count());

    caffe_gpu_set(blob->count(),float(0),blob->mutable_gpu_data());
    int num = blob->num();
    int channels = blob->channels();
    int height = blob->height();
    int width = blob->width();
    for (int n=0;n<num;n++)
    	for (int h=0;h<height;h++)
    		for (int w=0;w<width;w++)
    		{
    			int c = n;
    			blob->mutable_cpu_data()[((n*channels+c)*height+h)*width+w] = float(caffe_rng_rand()%3-1);
    		}
  }
};


class MSRAFiller : public Filler 
{
 public:
  explicit MSRAFiller(const FillerParameter& param) : Filler(param) 
  {}
  virtual void Fill(Blob* blob) 
  {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
   
    float std = sqrt(float(4) / float(fan_in + fan_out));

    //caffe_rng_uniform(blob->count(), -std*sqrt(3), std*sqrt(3), blob->mutable_cpu_data());
    caffe_rng_gaussian(blob->count(),0,std,blob->mutable_cpu_data());
  
  }	
};


class GlorotFiller : public Filler 
{
 public:
  explicit GlorotFiller(const FillerParameter& param) : Filler(param) 
  {}
  virtual void Fill(Blob* blob) 
  {
    CHECK(blob->count());
    int fan_in = blob->count() / blob->num();
    int fan_out = blob->count() / blob->channels();
    float std = sqrt(float(2) / float(fan_in + fan_out));
   caffe_rng_uniform(blob->count(), -std*sqrt(3), std*sqrt(3), blob->mutable_cpu_data());
  }
};


class GaussianFiller : public Filler {
 public:
  explicit GaussianFiller(const FillerParameter& param)
      : Filler(param) {}
  virtual void Fill(Blob* blob) 
  {
    float* data = blob->mutable_cpu_data();
    CHECK(blob->count());
    //caffe_rng_gaussian(blob->count(), float(this->filler_param_.mean()), float(this->filler_param_.std()), blob->mutable_cpu_data());
    float std = this->filler_param_.std();
    caffe_rng_uniform(blob->count(), -std*sqrt(3), std*sqrt(3), blob->mutable_cpu_data());
  }
};


class BilinearFiller : public Filler 
{
 public:
  explicit BilinearFiller(const FillerParameter& param) : Filler(param) 
  {}
  virtual void Fill(Blob* blob) {
    CHECK_EQ(blob->width(), blob->height()) << "Filter must be square";
    float* data = blob->mutable_cpu_data();
    int f = ceil(blob->width() / 2.);
    float c = (2 * f - 1 - f % 2) / (2. * f);
    for (int i = 0; i < blob->count(); ++i) 
    {
      float x = i % blob->width();
      float y = (i / blob->width()) % blob->height();
      data[i] = (1 - fabs(x / f - c)) * (1 - fabs(y / f - c));
    }
  }
};


class PottsFiller : public Filler 
{
 public:
  explicit PottsFiller(const FillerParameter& param): Filler(param) 
  {}
  virtual void Fill(Blob* blob) 
  {
    float* data = blob->mutable_cpu_data();
    caffe_set(blob->count(),float(0),data);
   	for(int i=0;i<blob->num();i++)
   		data[i+i*blob->channels()]=1;
  }
};

Filler* GetFiller(const FillerParameter& param);

}  // namespace caffe

#endif  // CAFFE_FILLER_HPP_
