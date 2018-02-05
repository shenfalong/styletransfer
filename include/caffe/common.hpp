#ifndef CAFFE_COMMON_HPP_
#define CAFFE_COMMON_HPP_

#include <boost/shared_ptr.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <nccl.h>
#include <climits>
#include <cmath>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>  // pair
#include <vector>
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/device_alternate.hpp"
#include "boost/random/mersenne_twister.hpp"
#include "boost/random/uniform_int.hpp"
#include "boost/thread.hpp"




#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)


// A simple macro to mark codes that are not implemented, so that when the code
// is executed we will see a fatal log.
#define NOT_IMPLEMENTED LOG(FATAL) << "Not Implemented Yet"

// See PR #1236
namespace cv { class Mat; }

namespace caffe {

typedef boost::mt19937 rng_t;

// We will use the boost shared_ptr instead of the new C++11 one mainly
// because cuda does not work (at least now) well with C++11 features.
using boost::shared_ptr;

// Common functions and classes from std that caffe often uses.
using std::fstream;
using std::ios;
using std::isnan;
using std::isinf;
using std::iterator;
using std::make_pair;
using std::map;
using std::ostringstream;
using std::pair;
using std::set;
using std::string;
using std::stringstream;
using std::vector;




#define GETSET(type,name)   \
	inline static void set_##name(type name) { Get().name##_ = name; } \
  inline static type name() { return Get().name##_; }
 

// A global initialization function that you should call in your main function.
// Currently it initializes google flags and google logging.
void GlobalInit(int* pargc, char*** pargv);

// A singleton class to hold common caffe stuff, such as the handler that
// caffe is going to use for cublas, curand, etc.
class Caffe {
 public:
  ~Caffe();

  // Thread local context for Caffe. Moved to common.cpp instead of
  // including boost/thread.hpp to avoid a boost/NVCC issues (#1009, #1010)
  // on OSX. Also fails on Linux with CUDA 7.0.18.
  inline static Caffe& Get() {
    if (!singleton_.get()) {
      singleton_.reset(new Caffe());
    }
    return *singleton_;
  }

	GETSET(string, drop_state)
  GETSET(string, bn_state)
  GETSET(string, gan_type)
  GETSET(string, gradient_penalty);
  GETSET(bool, reuse)
  GETSET(bool, frozen_param)
	GETSET(bool, second_pass)
	GETSET(vector<float *>, gpu_scalar)
	
	inline static rng_t* rng() { return Get().rng_; }
	

  static cublasHandle_t cublas_handle();
  static curandGenerator_t curand_generator();
  static cudnnHandle_t  cudnn_handle(int i);
  static ncclComm_t comms(int i);

  

	static vector<int> GPUs;
	static int number_collect_sample;


  static void SetDevice(const int device_id);
  
	static vector<void *> workspace_;
	static vector<void *> parallel_workspace_;
	static bool compare_variable(string var_name, string var_value);
	
	static boost::mutex mu_;  
	static boost::condition_variable_any cond_; 
	
	
 protected:
 	static shared_ptr<Caffe> singleton_;

 

  static vector<cublasHandle_t> cublas_handle_;
  static vector<curandGenerator_t> curand_generator_;
  
  static vector<cudnnHandle_t>  cudnn_handle_;
  
	static ncclComm_t* comms_;
	static vector<float *>gpu_scalar_;
	static string drop_state_;
	static string bn_state_;
	static string gan_type_;
	static string gradient_penalty_;
	static bool frozen_param_;
	static bool second_pass_;
	static bool reuse_;
	
	
	

	static rng_t * rng_;

 private:
  // The private constructor to avoid duplicate instantiation.
  Caffe();

  DISABLE_COPY_AND_ASSIGN(Caffe);
};


}  // namespace caffe

#endif  // CAFFE_COMMON_HPP_
