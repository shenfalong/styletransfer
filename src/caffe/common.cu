#include <boost/thread.hpp>
#include <glog/logging.h>
#include <cmath>
#include <cstdio>
#include <ctime>
#include "caffe/common.hpp"

namespace caffe {

//------------------static varibales---------------------
shared_ptr<Caffe> Caffe::singleton_;

std::vector<int> Caffe::GPUs;
int Caffe::number_collect_sample = -1;
vector<float *> Caffe::gpu_scalar_;

vector<void *> Caffe::workspace_;
vector<void *> Caffe::parallel_workspace_;

vector<cublasHandle_t> Caffe::cublas_handle_;
vector<curandGenerator_t> Caffe::curand_generator_;

boost::mutex Caffe::mu_;  
boost::condition_variable_any Caffe::cond_; 


vector<cudnnHandle_t>  Caffe::cudnn_handle_;
ncclComm_t* Caffe::comms_;

string Caffe::drop_state_ = "rand";
string Caffe::bn_state_ = "learned";
string Caffe::gan_type_ = "train_dnet";
string Caffe::gradient_penalty_ = "No";
bool Caffe::frozen_param_ = false;
bool Caffe::second_pass_ = false;
bool Caffe::reuse_ = false;

rng_t * Caffe::rng_;
//--------------------------------------------------------

void GlobalInit(int* pargc, char*** pargv) {
    // Google flags.
    ::gflags::ParseCommandLineFlags(pargc, pargv, true);
    // Google logging.
    ::google::InitGoogleLogging(*(pargv)[0]);
    // Provide a backtrace on segfault.
    ::google::InstallFailureSignalHandler();
}

// random seeding
int64_t cluster_seedgen(void) {
    int64_t s, seed, pid;
    FILE* f = fopen("/dev/urandom", "rb");
    if (f && fread(&seed, 1, sizeof(seed), f) == sizeof(seed)) {
        fclose(f);
        LOG(INFO)<<"random seed is "<<seed;
        return seed;
    }

    LOG(INFO) << "System entropy source not available, "
                 "using fallback algorithm to generate seed instead.";
    if (f)
        fclose(f);

    pid = getpid();
    s = time(NULL);
    seed = std::abs(((s * 181) * ((pid - 83) * 359)) % 104729);
    return seed;
}

//----------------------------------------
Caffe::Caffe() 
{
		LOG(INFO)<<"-----------set up Caffe enviroment----------";
		if (GPUs.size() == 0)
			LOG(FATAL)<<"------------Please set the GPU!!!!!!!!!!!!!!!----------------------------";
		CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));

		cudnn_handle_.resize(GPUs.size());
    cublas_handle_.resize(GPUs.size());
    curand_generator_.resize(GPUs.size());
	
		comms_ = (ncclComm_t*)malloc(sizeof(ncclComm_t)*Caffe::GPUs.size());
  	ncclCommInitAll(comms_,Caffe::GPUs.size(),Caffe::GPUs.data());
  		  
  	
    for(int i=0;i<GPUs.size();i++)
    {
      CUDA_CHECK(cudaSetDevice(Caffe::GPUs[i]));    	     	
			CUDNN_CHECK(cudnnCreate(&cudnn_handle_[i]));  
			cublasCreate(&cublas_handle_[i]);
			
			
			curandCreateGenerator(&curand_generator_[i], CURAND_RNG_PSEUDO_DEFAULT);
			curandSetPseudoRandomGeneratorSeed(curand_generator_[i], cluster_seedgen());	
    }
    CUDA_CHECK(cudaSetDevice(Caffe::GPUs[0]));
      
		rng_ = new rng_t(cluster_seedgen());
		
		gpu_scalar_.resize(NGPUS);
		for (int i=0;i<NGPUS;i++)		
			cudaMalloc((void**) &gpu_scalar_[i], sizeof(float));
}
Caffe::~Caffe() 
{
	//cudaFree(gpu_scalar);
//never delete static variables!!!!!!!!!!!!!!!!!
}
//----------------------------------------
bool Caffe::compare_variable(string var_name, string var_value)
{
	if (var_name == "gan_type")
		return gan_type() == var_value;	
	else if (var_name == "gradient_penalty")
		return gradient_penalty() == var_value;	
	else
	{
		LOG(FATAL)<<"unknown variable "<<var_name;
		return false;
	}
}

cublasHandle_t Caffe::cublas_handle()
{ 
	int d;
	CUDA_CHECK(cudaGetDevice(&d));
	int i;
	for(i=0;i<Caffe::GPUs.size();i++)
	{
		if(GPUs[i]==d)
	  	break;
	}
	if (i == Caffe::GPUs.size())
		LOG(FATAL)<<"wrong device "<<i;

	cublasHandle_t h = Get().cublas_handle_[i];
	return h;
}

cudnnHandle_t Caffe::cudnn_handle(int i)
{ 
	if(i >= Caffe::GPUs.size()*3)
		LOG(FATAL)<< "wrong device "<<i;
  cudnnHandle_t h = Get().cudnn_handle_[i];
  return h;
}
ncclComm_t Caffe::comms(int i)
{ 
	if(i >= Caffe::GPUs.size())
		LOG(FATAL)<< "wrong device"<<i;
  ncclComm_t c = Get().comms_[i];
  return c;
}

void Caffe::SetDevice(const int device_id) 
{
  int current_device;
  CUDA_CHECK(cudaGetDevice(&current_device));
  if (current_device == device_id) 
      return;
  
  // The call to cudaSetDevice must come before any calls to Get, which
  // may perform initialization using the GPU.
  CUDA_CHECK(cudaSetDevice(device_id));

  int i;
  for(i=0;i<Caffe::GPUs.size();i++)
  {
      if(GPUs[i]==device_id)
          break;
  }
  if (Get().cublas_handle_[i]) 
  		CUBLAS_CHECK(cublasDestroy(Get().cublas_handle_[i]));
  if (Get().curand_generator_[i]) 
  		CURAND_CHECK(curandDestroyGenerator(Get().curand_generator_[i]));

  CUBLAS_CHECK(cublasCreate(&Get().cublas_handle_[i]));
  CURAND_CHECK(curandCreateGenerator(&Get().curand_generator_[i],CURAND_RNG_PSEUDO_DEFAULT));
  CURAND_CHECK(curandSetPseudoRandomGeneratorSeed(Get().curand_generator_[i],cluster_seedgen()));
}
//----------------------------------------

curandGenerator_t Caffe::curand_generator() 
{
    int d;
    CUDA_CHECK(cudaGetDevice(&d));
    int i;
    for(i=0;i<Caffe::GPUs.size();i++)
    {
        if(GPUs[i]==d)
            break;
    }        
    if(i == Caffe::GPUs.size())
        LOG(FATAL)<< "wrong device"<<'\n';
    curandGenerator_t h = Get().curand_generator_[i];
    return h;
}

//----------------------------------------------------------------------------------------
const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
    case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}

const char* curandGetErrorString(curandStatus_t error) {
    switch (error) {
    case CURAND_STATUS_SUCCESS:
        return "CURAND_STATUS_SUCCESS";
    case CURAND_STATUS_VERSION_MISMATCH:
        return "CURAND_STATUS_VERSION_MISMATCH";
    case CURAND_STATUS_NOT_INITIALIZED:
        return "CURAND_STATUS_NOT_INITIALIZED";
    case CURAND_STATUS_ALLOCATION_FAILED:
        return "CURAND_STATUS_ALLOCATION_FAILED";
    case CURAND_STATUS_TYPE_ERROR:
        return "CURAND_STATUS_TYPE_ERROR";
    case CURAND_STATUS_OUT_OF_RANGE:
        return "CURAND_STATUS_OUT_OF_RANGE";
    case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
    case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case CURAND_STATUS_LAUNCH_FAILURE:
        return "CURAND_STATUS_LAUNCH_FAILURE";
    case CURAND_STATUS_PREEXISTING_FAILURE:
        return "CURAND_STATUS_PREEXISTING_FAILURE";
    case CURAND_STATUS_INITIALIZATION_FAILED:
        return "CURAND_STATUS_INITIALIZATION_FAILED";
    case CURAND_STATUS_ARCH_MISMATCH:
        return "CURAND_STATUS_ARCH_MISMATCH";
    case CURAND_STATUS_INTERNAL_ERROR:
        return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown curand status";
}

}  // namespace caffe
