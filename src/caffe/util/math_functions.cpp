#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <limits>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}



void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}


void caffe_axpy(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }



void caffe_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(float) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

void caffe_copy(const int N, const float* X, float* Y) {
	if (X != Y)
	{
		//CUDA_CHECK(cudaMemcpy(Y, X, sizeof(float) * N, cudaMemcpyDefault));
		caffe_gpu_memcpy(sizeof(float) * N,X,Y);
  }
}

void caffe_scal(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

void caffe_cpu_axpby(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}


void caffe_add(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

void caffe_sub(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

void caffe_mul(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

void caffe_div(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

void caffe_powx(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

void caffe_sqr(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}


void caffe_exp(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

void caffe_log(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

void caffe_abs(const int n, const float* a, float* y) {
    vsAbs(n, a, y);
}

int caffe_rng_rand() {
  int rand_num = (*Caffe::rng())();
  return int(abs(rand_num));
}


float caffe_nextafter(const float b) {
  return boost::math::nextafter<float>(
      b, std::numeric_limits<float>::max());
}




void caffe_rng_uniform(const int n, const float a, const float b, float* r) {
	CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(a, b);
  boost::uniform_real<float > random_distribution(a, caffe_nextafter(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<float > >
      variate_generator(Caffe::rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}



void caffe_rng_gaussian(const int n, const float a, const float sigma, float* r) 
{
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(sigma, 0);
  boost::normal_distribution<float > random_distribution(a, sigma);
  boost::variate_generator<caffe::rng_t*, boost::normal_distribution<float > >
      variate_generator(Caffe::rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}




void caffe_rng_bernoulli(const int n, const float p, int* r) 
{
	CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(p, 0);
  CHECK_LE(p, 1);
  boost::bernoulli_distribution<float > random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<float > >
      variate_generator(Caffe::rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}


void caffe_rng_bernoulli(const int n, const float p, unsigned int* r) {
//TODO
NOT_IMPLEMENTED;
}




float caffe_cpu_strided_dot(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}


float caffe_cpu_dot(const int n, const float* x, const float* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}





float caffe_cpu_asum(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}


void caffe_cpu_scale(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}



}  // namespace caffe
