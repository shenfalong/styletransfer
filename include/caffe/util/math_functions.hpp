#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit

#include "glog/logging.h"

#include "caffe/common.hpp"
#include "caffe/util/device_alternate.hpp"
#include "caffe/util/mkl_alternate.hpp"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.

void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);


void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y);


void caffe_axpy(const int N, const float alpha, const float* X,
    float* Y);


void caffe_cpu_axpby(const int N, const float alpha, const float* X,
    const float beta, float* Y);


void caffe_copy(const int N, const float *X, float *Y);


void caffe_set(const int N, const float alpha, float *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}


void caffe_add_scalar(const int N, const float alpha, float *X);


void caffe_scal(const int N, const float alpha, float *X);


void caffe_sqr(const int N, const float* a, float* y);


void caffe_add(const int N, const float* a, const float* b, float* y);


void caffe_sub(const int N, const float* a, const float* b, float* y);


void caffe_mul(const int N, const float* a, const float* b, float* y);


void caffe_div(const int N, const float* a, const float* b, float* y);


void caffe_powx(const int n, const float* a, const float b, float* y);

int caffe_rng_rand();


float caffe_nextafter(const float b);


void caffe_rng_uniform(const int n, const float a, const float b, float* r);


void caffe_rng_gaussian(const int n, const float mu, const float sigma,
                        float* r);


void caffe_rng_bernoulli(const int n, const float p, int* r);


void caffe_rng_bernoulli(const int n, const float p, unsigned int* r);


void caffe_exp(const int n, const float* a, float* y);


void caffe_log(const int n, const float* a, float* y);


void caffe_abs(const int n, const float* a, float* y);


float caffe_cpu_dot(const int n, const float* x, const float* y);


float caffe_cpu_strided_dot(const int n, const float* x, const int incx,
    const float* y, const int incy);

// Returns the sum of the absolute values of the elements of vector x

float caffe_cpu_asum(const int n, const float* x);


void caffe_cpu_scale(const int n, const float alpha, const float *x, float* y);

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.

void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C);


void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const float alpha, const float* A, const float* x, const float beta,
    float* y);


void caffe_gpu_axpy(const int N, const float alpha, const float* X,
    float* Y);


void caffe_gpu_axpby(const int N, const float alpha, const float* X,
    const float beta, float* Y);

void caffe_gpu_memcpy(const size_t N, const void *X, void *Y);


void caffe_gpu_set(const int N, const float alpha, float *X);

inline void caffe_gpu_memset(const size_t N, const int alpha, void* X) {
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemset(X, alpha, N));  // NOLINT(caffe/alt_fn)
#else
  NO_GPU;
#endif
}


void caffe_gpu_add_scalar(const int N, const float alpha, float *X);


void caffe_gpu_scal(const int N, const float alpha, float *X);


void caffe_gpu_add(const int N, const float alpha, const float* a, const float beta, const float* b, float* y);


void caffe_gpu_sub(const int N, const float* a, const float* b, float* y);


void caffe_gpu_mul(const int N, const float* a, const float* b, float* y);


void caffe_gpu_add_mul(const int N, const float* a, const float* b, float* y);


void caffe_gpu_div(const int N, const float* a, const float* b, float* y);


void caffe_gpu_abs(const int n, const float* a, float* y);


void caffe_gpu_exp(const int n, const float* a, float* y);


void caffe_gpu_log(const int n, const float* a, float* y);


void caffe_gpu_powx(const int n, const float* a, const float b, float* y);

// caffe_gpu_rng_uniform with two arguments generates integers in the range
// [0, UINT_MAX].
void caffe_gpu_rng_uniform(const int n, unsigned int* r);

// caffe_gpu_rng_uniform with four arguments generates floats in the range
// (a, b] (strictly greater than a, less than or equal to b) due to the
// specification of curandGenerateUniform.  With a = 0, b = 1, just calls
// curandGenerateUniform; with other limits will shift and scale the outputs
// appropriately after calling curandGenerateUniform.

void caffe_gpu_rng_uniform(const int n, const float a, const float b, float* r);


void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma,
                            float* r);


void caffe_gpu_rng_bernoulli(const int n, const float p, int* r);


void caffe_gpu_dot(const int n, const float* x, const float* y, float* out);


void caffe_gpu_asum(const int n, const float* x, float* y);


void caffe_gpu_sign(const int n, const float* x, float* y);


void caffe_gpu_sgnbit(const int n, const float* x, float* y);


void caffe_gpu_fabs(const int n, const float* x, float* y);


void caffe_gpu_scale(const int n, const float alpha, const float *x, float* y);


void box_filter_gpu(const int num, const int channels, const int height,const int width, const int radius, const float *id, float *od,float * buffer);


void adam_update_gpu(int N, float* g, float* m, float* v, float beta1,
    float beta2, float eps_hat, float corrected_local_rate);


float caffe_gpu_sum(const int N, const float *in);

float caffe_gpu_square_sum(const int N, const float *in);

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
