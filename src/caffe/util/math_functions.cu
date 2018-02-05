#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit
#include <thrust/device_vector.h>
#include <thrust/functional.h>  // thrust::plus
#include <thrust/reduce.h>

#include <cmath>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void caffe_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;  
     
  CUBLAS_CHECK(cublasSgemm(Caffe::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));  
   
}


void caffe_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  CUBLAS_CHECK(cublasSgemv(Caffe::cublas_handle(), cuTransA, N, M, &alpha,
     A, N, x, 1, &beta, y, 1)); 
}


void caffe_gpu_axpy(const int N, const float alpha, const float* X,
    float* Y) {
  CUBLAS_CHECK(cublasSaxpy(Caffe::cublas_handle(), N, &alpha, X, 1, Y, 1));
}



void caffe_gpu_memcpy(const size_t N, const void* X, void* Y) {
  if (X != Y) {
    CUDA_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
  }
}


void caffe_gpu_scal(const int N, const float alpha, float *X) {
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), N, &alpha, X, 1));
}




void caffe_gpu_axpby(const int N, const float alpha, const float* X,
    const float beta, float* Y) {
  caffe_gpu_scal(N, beta, Y);
  caffe_gpu_axpy(N, alpha, X, Y);
}



void caffe_gpu_dot(const int n, const float* x, const float* y,
    float* out) {
  CUBLAS_CHECK(cublasSdot(Caffe::cublas_handle(), n, x, 1, y, 1, out));
}




void caffe_gpu_asum(const int n, const float* x, float* y) {
  CUBLAS_CHECK(cublasSasum(Caffe::cublas_handle(), n, x, 1, y));
}




void caffe_gpu_scale(const int n, const float alpha, const float *x,
                            float* y) {
  CUBLAS_CHECK(cublasScopy(Caffe::cublas_handle(), n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(Caffe::cublas_handle(), n, &alpha, y, 1));
}




__global__ void set_kernel(const int n, const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}


void caffe_gpu_set(const int N, const float alpha, float* Y) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(float) * N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}




__global__ void add_scalar_kernel(const int n, const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}


void caffe_gpu_add_scalar(const int N, const float alpha, float* Y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, alpha, Y);
}




__global__ void add_kernel(const int n, const float alpha, const float* a, const float beta, const float* b, 
																																															float* y) 
{
  CUDA_KERNEL_LOOP(index, n) 
  {
    y[index] = alpha * a[index] + beta * b[index];
  }
}


void caffe_gpu_add(const int N, const float alpha, const float* a, const float beta, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
  (N, alpha, a, beta, b, y);
}



__global__ void sub_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}


void caffe_gpu_sub(const int N, const float* a, const float* b,
    float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}




__global__ void mul_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}


void caffe_gpu_mul(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}



__global__ void add_mul_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = y[index] + a[index] * b[index];
  }
}


void caffe_gpu_add_mul(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_mul_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}




__global__ void div_kernel(const int n, const float* a,
    const float* b, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}


void caffe_gpu_div(const int N, const float* a,
    const float* b, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}




__global__ void abs_kernel(const int n, const float* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}


void caffe_gpu_abs(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}





__global__ void exp_kernel(const int n, const float* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}


void caffe_gpu_exp(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}




__global__ void log_kernel(const int n, const float* a, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = log(a[index]);
  }
}


void caffe_gpu_log(const int N, const float* a, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  log_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, y);
}



__global__ void powx_kernel(const int n, const float* a,
    const float alpha, float* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}


void caffe_gpu_powx(const int N, const float* a,
    const float alpha, float* y) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, a, alpha, y);
}



void caffe_gpu_rng_uniform(const int n, unsigned int* r) {
  CURAND_CHECK(curandGenerate(Caffe::curand_generator(), r, n));
}


void caffe_gpu_rng_uniform(const int n, const float a, const float b, float* r) {
  CURAND_CHECK(curandGenerateUniform(Caffe::curand_generator(), r, n));
  const float range = b - a;
  if (range != static_cast<float>(1)) {
    caffe_gpu_scal(n, range, r);
  }
  if (a != static_cast<float>(0)) {
    caffe_gpu_add_scalar(n, a, r);
  }
}


void caffe_gpu_rng_gaussian(const int n, const float mu, const float sigma, float* r) {
  CURAND_CHECK(
      curandGenerateNormal(Caffe::curand_generator(), r, n, mu, sigma));
}




__global__ void box_filter_x_kernel(const int num, const int channels, const int height, const int width,int radius, const float * id, float *od) {
	CUDA_KERNEL_LOOP(ind, height*channels*num)
	{
		float sum=0;
		for (int w = 0; w <= min(radius,width-1); w++)
			sum += id[ind*width+w];
		od[ind*width+0] = sum;
		for (int w = 1; w < width-radius; w++)
		{
			sum += id[ind*width+w+radius];
			if(w-radius > 0)
				sum -= id[ind*width+w-radius-1];
			od[ind*width+w] = sum;
		}
		for (int w = max(width-radius,1); w < width; w++)
		{
			if(w-radius > 0)
				sum -= id[ind*width+w-radius-1];
			od[ind*width+w] = sum;
		}
	}
}

__global__ void box_filter_y_kernel(const int num,const int channels, const int height,const int width, const int radius,const float area, const float * id, float *od) {
	CUDA_KERNEL_LOOP(ind, width*channels*num)
	{
		int c=ind / width;
		int w=ind % width;
		float sum=0;
		for (int h = 0; h <= min(radius,height-1); h++)
			sum += id[(c*height+h)*width+w];
		od[(c*height+0)*width+w] = sum / area;
		for (int h = 1; h < height-radius; h++)
		{
			sum += id[(c*height+h+radius)*width+w];
			if(h-radius > 0)
				sum -= id[(c*height+h-radius-1)*width+w];
			od[(c*height+h)*width+w] = sum / area;
		}
		for (int h= max(height - radius,1); h < height; h++)
		{
			if(h-radius > 0)
				sum -= id[(c*height+h-radius-1)*width+w];
			od[(c*height+h)*width+w] = sum / area;
		}
	}
}


void box_filter_gpu(const int num, const int channels, const int height,const int width, const int radius, const float *id, float *od,float * buffer) {
	box_filter_x_kernel<<<CAFFE_GET_BLOCKS(height*channels*num), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels, height,width, radius, id, buffer);
	box_filter_y_kernel<<<CAFFE_GET_BLOCKS(width*channels*num), CAFFE_CUDA_NUM_THREADS>>>
  (num,channels, height,width, radius,min((2*radius+1),height) * min((2*radius+1),width), buffer, od);
}



__global__ void AdamUpdate(int N, float* g, float* m, float* v, float beta1, float beta2, float eps_hat, float corrected_local_rate) {
  CUDA_KERNEL_LOOP(i, N) 
  {
    float gi = g[i];
    float mi = m[i] = m[i]*beta1 + gi*(1-beta1);
    float vi = v[i] = v[i]*beta2 + gi*gi*(1-beta2);
    g[i] = corrected_local_rate * mi / (sqrt(vi) + eps_hat);
  }
}

void adam_update_gpu(int N, float* g, float* m, float* v, float beta1,
    float beta2, float eps_hat, float corrected_local_rate) {
  AdamUpdate <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>
  (N, g, m, v, beta1, beta2, eps_hat, corrected_local_rate);
  CUDA_POST_KERNEL_CHECK;
}

    

__global__ void sum_kernel(int N, const float* in, float* sum) 
{
	__shared__ float buffer[CAFFE_CUDA_NUM_THREADS];
	buffer[threadIdx.x]=0;
 	for (int i = threadIdx.x; i < N; i += blockDim.x)
 		buffer[threadIdx.x] += in[i];
 	__syncthreads();

 	for (int s = blockDim.x/2; s > 0; s >>= 1)
 	{
 		if (threadIdx.x < s) 
 			buffer[threadIdx.x] += buffer[threadIdx.x+s];
 		__syncthreads();
 	}
	
 	if (threadIdx.x == 0)
 		 sum[0] = buffer[0];
}    

float caffe_gpu_sum(const int N, const float *in)
{
	int gpu_id_;
  CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
	
	float cpu_sum;
  sum_kernel<<<1, CAFFE_CUDA_NUM_THREADS>>>
  (N, in, Caffe::gpu_scalar()[gpu_id_]);
  CUDA_CHECK(cudaMemcpy(&cpu_sum, Caffe::gpu_scalar()[gpu_id_], sizeof(float), cudaMemcpyDeviceToHost));
  return cpu_sum;
}
__global__ void square_sum_kernel(int N, const float* in, float* sum) 
{
	__shared__ float buffer[CAFFE_CUDA_NUM_THREADS];
	buffer[threadIdx.x]=0;
 	for (int i = threadIdx.x; i < N; i += blockDim.x)
 		buffer[threadIdx.x] += in[i]*in[i];
 	__syncthreads();

 	for (int s = blockDim.x/2; s > 0; s >>= 1)
 	{
 		if (threadIdx.x < s) 
 			buffer[threadIdx.x] += buffer[threadIdx.x+s];
 		__syncthreads();
 	}
	
 	if (threadIdx.x == 0)
 		 sum[0] = buffer[0];
}    

float caffe_gpu_square_sum(const int N, const float *in)
{
	int gpu_id_;
  CUDA_CHECK(cudaGetDevice(&gpu_id_));
	int i;
	for (i=0;i<Caffe::GPUs.size();i++)
		if (Caffe::GPUs[i] == gpu_id_)
			break;
	gpu_id_ = i;
  
	float cpu_sum;
  square_sum_kernel<<<1, CAFFE_CUDA_NUM_THREADS>>>
  (N, in, Caffe::gpu_scalar()[gpu_id_]);
  CUDA_CHECK(cudaMemcpy(&cpu_sum, Caffe::gpu_scalar()[gpu_id_], sizeof(float), cudaMemcpyDeviceToHost));
  return cpu_sum;
}
}  // namespace caffe
