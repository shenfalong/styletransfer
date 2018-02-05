#include <algorithm>
#include <cfloat>
#include <vector>

#include "thrust/device_vector.h"

#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/layers/operator/guided_crf_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

static __global__ void softmax_forward_kernel(const int maxStates,const int nNodes, const float * energy,float * prob)
{
	CUDA_KERNEL_LOOP(n, nNodes)
	{
		for(int s=0;s<maxStates;s++)
			prob[s*nNodes+n] = energy[s*nNodes+n];

		float max_prob = float(-FLT_MAX);
		for(int s=0;s<maxStates;s++)
			max_prob =max(max_prob,prob[s*nNodes+n]);

		for(int s=0;s<maxStates;s++)
			prob[s*nNodes+n] -= max_prob;

		float sum = 0;
		for(int s=0;s<maxStates;s++)
			sum += exp(prob[s*nNodes+n]);

		for(int s=0;s<maxStates;s++)
			prob[s*nNodes+n] = exp(prob[s*nNodes+n]) / sum;
	}
}

static __global__ void softmax_backward_kernel(const int maxStates,const int nNodes, const float * top_diff,const float *prob,float * bottom_diff)
{
	CUDA_KERNEL_LOOP(ind, nNodes*maxStates)
	{
		int n=ind % nNodes;
		int s=ind / nNodes;
		float sum = 0;
		for(int s2=0;s2<maxStates;s2++)
			 sum += top_diff[s2*nNodes+n]*prob[s2*nNodes+n]*(float(s==s2)-prob[s*nNodes+n]);
		bottom_diff[s*nNodes+n] = sum;
	}
}
//--------------------------------------------------------------

static __global__ void vector_product_kernel(const int num,const int channels1,const int channels2, const int spatial_dim,const float * a,const float * b,float *var)//var = a .* b
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*channels1*channels2*num)
	{	
		int n   = ind / spatial_dim / channels1 / channels2;
		int c2  = ind / spatial_dim / channels1 % channels2;
		int c1  = ind / spatial_dim % channels1;
		int s   = ind % spatial_dim;
		
		
		var[ind]=a[(n*channels1+c1)*spatial_dim+s]*b[(n*channels2+c2)*spatial_dim+s];
	}
}

static __global__ void substract_vector_product_kernel(const int num, const int channels1,const int channels2,const int spatial_dim,const float *avg,const float *a,const float *b, float * var)//var = avg - a.*b;
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*channels1*channels2*num)
	{
		int n  = ind / spatial_dim / channels1 / channels2;
		int c2 = ind / spatial_dim / channels1 % channels2;
		int c1 = ind / spatial_dim % channels1;	
		int s  = ind % spatial_dim;
		var[ind]=avg[ind]-a[(n*channels1+c1)*spatial_dim+s]*b[(n*channels2+c2)*spatial_dim+s];
	}
}

static __global__ void inv_var_I_eps_kernel_3(const int num, const int channels, const int spatial_dim, const float eps,float *var_I,float *inv_var_I)
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*num)
	{
		int n = ind / spatial_dim;
		int s = ind % spatial_dim;
		
		for(int c=0;c<channels;c++)
			var_I[(n*channels*channels+(c*channels+c))*spatial_dim+s]=var_I[(n*channels*channels+(c*channels+c))*spatial_dim+s]+eps;

		float det = var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*(var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]-var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s])
				- var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s])
				+ var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]);

		inv_var_I[(n*channels*channels+0*channels+0)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+0*channels+1)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+0*channels+2)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]
				-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]);

		inv_var_I[(n*channels*channels+1*channels+0)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+1*channels+1)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+1*channels+2)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]);

		inv_var_I[(n*channels*channels+2*channels+0)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+2*channels+1)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+0*channels+2)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+2)*spatial_dim+s]);
		inv_var_I[(n*channels*channels+2*channels+2)*spatial_dim+s] = 1/det*(var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]
				-var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]*var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]);
	}

}

static __global__ void div_sum_kernel_3(const int num, const int channels,const int maxStates,const int spatial_dim,const float *inv_var_I,const float *cov_Ip,
																 float *a)
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*maxStates*num)
	{
		int n = ind / spatial_dim / maxStates;
		int m = ind / spatial_dim % maxStates;
		int s = ind % spatial_dim;
		
		
		a[((n*maxStates+m)*channels+0)*spatial_dim+s] = cov_Ip[((n*maxStates+m)*channels+0)*spatial_dim+s]*inv_var_I[(n*channels*channels+0*channels+0)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+1)*spatial_dim+s]*inv_var_I[(n*channels*channels+0*channels+1)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+2)*spatial_dim+s]*inv_var_I[(n*channels*channels+0*channels+2)*spatial_dim+s];

		a[((n*maxStates+m)*channels+1)*spatial_dim+s]	= cov_Ip[((n*maxStates+m)*channels+0)*spatial_dim+s]*inv_var_I[(n*channels*channels+1*channels+0)*spatial_dim+s]
																	  + cov_Ip[((n*maxStates+m)*channels+1)*spatial_dim+s]*inv_var_I[(n*channels*channels+1*channels+1)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+2)*spatial_dim+s]*inv_var_I[(n*channels*channels+1*channels+2)*spatial_dim+s];

		a[((n*maxStates+m)*channels+2)*spatial_dim+s] = cov_Ip[((n*maxStates+m)*channels+0)*spatial_dim+s]*inv_var_I[(n*channels*channels+2*channels+0)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+1)*spatial_dim+s]*inv_var_I[(n*channels*channels+2*channels+1)*spatial_dim+s]
																		+ cov_Ip[((n*maxStates+m)*channels+2)*spatial_dim+s]*inv_var_I[(n*channels*channels+2*channels+2)*spatial_dim+s];
	}
}

static __global__ void substract_vector_matrix_product_kernel_3(const int num, const int channels,const int maxStates,const int spatial_dim,const float * mean_p,const float * a,const float * mean_I,float *b)//	b = mean_p - mean_I *. a;
{
	CUDA_KERNEL_LOOP(ind, spatial_dim*maxStates*num)
	{
		int n = ind / spatial_dim / maxStates;
		int m = ind / spatial_dim % maxStates;
		int s = ind % spatial_dim;
		
		b[ind] = mean_p[ind]
				   - mean_I[(n*3+0)*spatial_dim+s] * a[((n*maxStates+m)*channels+0)*spatial_dim+s]
				   - mean_I[(n*3+1)*spatial_dim+s] * a[((n*maxStates+m)*channels+1)*spatial_dim+s]
				   - mean_I[(n*3+2)*spatial_dim+s] * a[((n*maxStates+m)*channels+2)*spatial_dim+s];
	}
}

static __global__ void vector_matrix_product_sum_kernel_3(const int num, const int channels,const int maxStates,const int spatial_dim,const float *mean_a,const float *I,const float *mean_b,float *q)// q = I .* mean_a + mean_b;
{

	CUDA_KERNEL_LOOP(ind, spatial_dim*maxStates*num)
	{
		int n = ind / spatial_dim / maxStates;
		int m = ind / spatial_dim % maxStates;
		int s = ind % spatial_dim;
		
		q[ind] = I[(n*3+0)*spatial_dim+s] * mean_a[((n*maxStates+m)*channels+0)*spatial_dim+s]
					 + I[(n*3+1)*spatial_dim+s] * mean_a[((n*maxStates+m)*channels+1)*spatial_dim+s]
				   + I[(n*3+2)*spatial_dim+s] * mean_a[((n*maxStates+m)*channels+2)*spatial_dim+s]
				   + mean_b[ind];
	}

}
//---------------------------------------------

void GuidedCRFLayer::guided_filter_gpu(const int num,const int channels,const int maxStates,const int height,const int width,const float *I,const float * p,float *output_p)
{
	const int spatial_dim=height*width;

	//******************************** prob ************************************
	box_filter_gpu(num,maxStates,height,width,radius,p,mean_p.mutable_gpu_data(),buffer_score.mutable_gpu_data());

	vector_product_kernel<<<CAFFE_GET_BLOCKS(num*channels*maxStates*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim,I,p,Ip);//Ip = I .* p;
	
	box_filter_gpu(num,channels*maxStates,height,width,radius,Ip,mean_Ip,buffer_image_score);


	substract_vector_product_kernel<<<CAFFE_GET_BLOCKS(num*channels*maxStates*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim,mean_Ip,mean_I.gpu_data(),mean_p.gpu_data(), cov_Ip);//cov_Ip = mean_Ip - mean_I .* mean_p;


	inv_var_I_eps_kernel_3<<<CAFFE_GET_BLOCKS(num*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,spatial_dim,eps,var_I.mutable_gpu_data(),inv_var_I.mutable_gpu_data());//inv_var_I=inv(var_I + eps);


	div_sum_kernel_3<<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim,inv_var_I.gpu_data(),cov_Ip,a);//a = cov_Ip ./ inv_var_I;

	box_filter_gpu(num,channels*maxStates,height,width,radius,a,mean_a,buffer_image_score);

	substract_vector_matrix_product_kernel_3<<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
  (num,channels,maxStates,spatial_dim,mean_p.gpu_data(),a,mean_I.gpu_data(),b.mutable_gpu_data());//	b = mean_p - mean_I .* a;


	box_filter_gpu(num,maxStates,height,width,radius,b.gpu_data(),mean_b.mutable_gpu_data(),buffer_score.mutable_gpu_data());

	vector_matrix_product_sum_kernel_3<<<CAFFE_GET_BLOCKS(num*maxStates*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,maxStates,spatial_dim,mean_a,I,mean_b.gpu_data(),output_p);// q = I .* mean_a + mean_b;

}


void GuidedCRFLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
	const float * nodePot = bottom[0]->gpu_data();
	const float * imageData = bottom[1]->gpu_data();


	int num = bottom[0]->num();
	int maxStates = bottom[0]->channels();
	int channels = bottom[1]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int spatial_dim=height*width;

	int nNodes = num*width *height;
	

	//******************************** image ************************************
	box_filter_gpu(num,channels,height,width,radius,imageData,mean_I.mutable_gpu_data(),buffer_image.mutable_gpu_data());

	vector_product_kernel<<<CAFFE_GET_BLOCKS(num*channels*channels*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,channels,spatial_dim,imageData,imageData,II.mutable_gpu_data());// II = I .* I;

	box_filter_gpu(num,channels*channels,height,width,radius,II.gpu_data(),mean_II.mutable_gpu_data(),buffer_image_image.mutable_gpu_data());

	substract_vector_product_kernel<<<CAFFE_GET_BLOCKS(num*channels*channels*spatial_dim), CAFFE_CUDA_NUM_THREADS>>>
	(num,channels,channels,spatial_dim,mean_II.gpu_data(),mean_I.gpu_data(),mean_I.gpu_data(), var_I.mutable_gpu_data());//var_I = mean_II - mean_I .* mean_I;
	//-----------------------------------------------------------------------------------


	caffe_copy(tempPot.count(),nodePot,tempPot.mutable_gpu_data());
	for(int iter = 0; iter < maxIter; iter++)
	{
		softmax_forward_kernel<<<CAFFE_GET_BLOCKS(nNodes), CAFFE_CUDA_NUM_THREADS>>>
		(maxStates,nNodes,tempPot.gpu_data(),nodeBel[iter]->mutable_gpu_data());


		guided_filter_gpu(num,channels,maxStates,height,width,imageData,nodeBel[iter]->gpu_data(),filterPot.mutable_gpu_data());
	

		caffe_gpu_gemm(CblasNoTrans, CblasNoTrans, maxStates, nNodes, maxStates,
													(float)1., this->blobs_[0]->gpu_data(), filterPot.gpu_data(),
													(float)0., compatPot.mutable_gpu_data());

		caffe_gpu_add(maxStates*nNodes,float(1),nodePot,alpha,compatPot.gpu_data(),tempPot.mutable_gpu_data());
	}
	caffe_copy(top[0]->count(),tempPot.gpu_data(),top[0]->mutable_gpu_data());
}

void GuidedCRFLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom)
{
	int num = bottom[0]->num();
	int maxStates = bottom[0]->channels();
	int channels = bottom[1]->channels();
	int height = bottom[0]->height();
	int width = bottom[0]->width();
	int nNodes = num*width *height;

//----------------------- workspace -------------------------
	myworkspace_[0]->Reshape(num*maxStates,channels,height,width);
	myworkspace_[1]->Reshape(num*maxStates,channels,height,width);
	myworkspace_[2]->Reshape(num*maxStates,channels,height,width);
	
  Ip = myworkspace_[0]->mutable_gpu_data();
  mean_Ip = myworkspace_[0]->mutable_gpu_diff();
  
  cov_Ip = myworkspace_[1]->mutable_gpu_data();
  a = myworkspace_[1]->mutable_gpu_diff();
  
  mean_a = myworkspace_[2]->mutable_gpu_data();
  buffer_image_score  = myworkspace_[2]->mutable_gpu_diff();
//---------------------------------------------------------------- 


	const float *top_diff = top[0]->gpu_diff();
	float * bottom_diff = bottom[0]->mutable_gpu_diff();

	const float * imageData = bottom[1]->gpu_data();
	

	caffe_gpu_set(filterPot.count(),float(0),filterPot.mutable_gpu_diff());
	caffe_gpu_set(compatPot.count(),float(0),compatPot.mutable_gpu_diff());
	caffe_gpu_set(tempPot.count(),float(0),tempPot.mutable_gpu_diff());
	caffe_gpu_set(bottom[0]->count(),float(0),bottom_diff);



	caffe_copy(tempPot.count(),top_diff,tempPot.mutable_gpu_diff());
	
	for(int iter = maxIter-1; iter >= 0; iter--)
	{
		caffe_gpu_add(maxStates*nNodes,alpha,tempPot.gpu_diff(),float(0),compatPot.gpu_diff(),compatPot.mutable_gpu_diff());
		caffe_gpu_add(maxStates*nNodes,float(1) ,tempPot.gpu_diff(),float(1),bottom_diff         ,bottom_diff);


		caffe_gpu_gemm(CblasTrans, CblasNoTrans, maxStates, nNodes, maxStates,
													(float)1., this->blobs_[0]->gpu_data(), compatPot.gpu_diff(),
													(float)0., filterPot.mutable_gpu_diff());

		guided_filter_gpu(num,channels,maxStates,height,width,imageData,filterPot.gpu_diff(),nodeBel[iter]->mutable_gpu_diff());

		softmax_backward_kernel<<<CAFFE_GET_BLOCKS(maxStates*nNodes), CAFFE_CUDA_NUM_THREADS>>>
		(maxStates,nNodes,nodeBel[iter]->gpu_diff(),nodeBel[iter]->gpu_data(),tempPot.mutable_gpu_diff());
	}	
	caffe_gpu_add(tempPot.count(),float(1),tempPot.gpu_diff(),float(1),bottom[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
}

void GuidedCRFLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top)
{
}

}  // namespace caffe
