#ifndef CAFFE_BLOB_HPP_
#define CAFFE_BLOB_HPP_

#include <algorithm>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/syncedmem.hpp"

const int kMaxBlobAxes = 32;

namespace caffe {


class Blob 
{
 public:
  Blob() : data_(), diff_(),sec_diff_(), count_(0), capacity_(0) {}
	explicit Blob(const int num, const int channels, const int height, const int width) : capacity_(0)
	{
		Reshape(num, channels, height, width);
	}



  void Reshape(const int num, const int channels, const int height, const int width);
	void FromProto(const BlobProto& proto);
  void ToProto(BlobProto* proto, bool write_diff = false) const;
  
  void set_data(Blob& other);
  void set_diff(Blob& other); 
  void set_sec_diff(Blob& other); 
  
  
  
  
	void ReshapeLike(const Blob& other) { Reshape(other.num(),other.channels(),other.height(),other.width()); }
  inline int count() const { return count_; }
	inline int num() const { return num_; }
	inline int channels() const { return channels_; }
	inline int height() const { return height_; }
	inline int width() const { return width_; }
  inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const  { return ((n * channels() + c) * height() + h) * width() + w; }
	const float* cpu_data() const  { return (const float*)data_->cpu_data(); }
	const float* gpu_data() const  { return (const float*)data_->gpu_data(); }
	const float* cpu_diff() const  { return (const float*)diff_->cpu_data(); }
	const float* gpu_diff() const  { return (const float*)diff_->gpu_data(); }
	const float* cpu_sec_diff() const  { return (const float*)sec_diff_->cpu_data(); }
	const float* gpu_sec_diff() const  { return (const float*)sec_diff_->gpu_data(); }
	float* mutable_cpu_data() { return static_cast<float*>(data_->mutable_cpu_data()); }
	float* mutable_gpu_data() { return static_cast<float*>(data_->mutable_gpu_data()); }
	float* mutable_cpu_diff() { return static_cast<float*>(diff_->mutable_cpu_data()); }
	float* mutable_gpu_diff() { return static_cast<float*>(diff_->mutable_gpu_data()); }
	float* mutable_cpu_sec_diff() { return static_cast<float*>(sec_diff_->mutable_cpu_data()); }
	float* mutable_gpu_sec_diff() { return static_cast<float*>(sec_diff_->mutable_gpu_data()); }
	inline const shared_ptr<SyncedMemory>& data() const  { return data_; }
  inline const shared_ptr<SyncedMemory>& diff() const  { return diff_; }
  inline const shared_ptr<SyncedMemory>& sec_diff() const  { return sec_diff_; }
  
  void set_cpu_data(float* data) { data_->set_cpu_data(data); }
  
 protected:
  shared_ptr<SyncedMemory> data_;
  shared_ptr<SyncedMemory> diff_;
  shared_ptr<SyncedMemory> sec_diff_;
  int count_;
  int capacity_;
	int num_;
	int channels_;
	int height_;
	int width_;

  DISABLE_COPY_AND_ASSIGN(Blob);
};  // class Blob

}  // namespace caffe

#endif  // CAFFE_BLOB_HPP_
