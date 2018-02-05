#include <vector>

#include "caffe/layers/activation/imresize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



static __global__ void resize_2_kernel(int count,int stride,int kernel_size,int channels,int height, int width,const float * in, float * out_0, float *out_1)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height;
  	int h = index / width % height;
  	int w = index % width;
  	
  	int pad = (kernel_size - 1)/ 2;
  	
  	int hstart = max(h * stride - pad,0);
  	int hend = min(h * stride  - pad + kernel_size,height*stride);
  	int wstart = max(w * stride - pad,0);
  	int wend = min(w * stride  -pad + kernel_size,width*stride);
  	
  	float hist[21];
  	for(int c=0;c<channels;c++)	
  		hist[c] = 0;
  	for(int ih = hstart;ih<hend;ih++)
  		for(int iw = wstart;iw<wend;iw++)
  		{ 	
  			int cur_ind = (n * stride*height + ih) * stride*width + iw;  						
  			if (in[cur_ind] >= 0 && in[cur_ind] < channels)// never use != , please try < or > instread...
  				hist[int(in[cur_ind])] += (1 - abs(float(ih - (h * stride + 0.5)))/float(stride)) * (1 - abs(float(iw - (w * stride + 0.5)))/float(stride));
  		}
  	float max_value = -1;
  //	float max_index = -1;
  	float sum = 0;
  	for(int c=0;c<channels;c++)
  	{
  		sum += hist[c];
  		if (hist[c]>max_value)
  		{
  			max_value = hist[c];
  	//		max_index = c;
  		}
  	}		
  	if(max_value > 0)
  	{
  		for (int c=0;c<channels;c++)
  			out_0[((n*channels+c)*height+h)*width+w] = hist[c] / sum;		
  		out_1[index] = 1;
  		
  	}
  	else
  	{
  		for (int c=0;c<channels;c++)
  			out_0[((n*channels+c)*height+h)*width+w] = 0;
  		out_1[index] = 0; 		
  	}	
  }
}    


static __global__ void resize_1_kernel(int count,int stride,int kernel_size,int channels,int height, int width,const float * in, float * out_0)
{
  CUDA_KERNEL_LOOP(index, count)
  {
  	int n = index / width / height;
  	int h = index / width % height;
  	int w = index % width;
  	
  	int pad = (kernel_size - 1)/ 2;
  	
  	int hstart = max(h * stride - pad,0);
  	int hend = min(h * stride  - pad + kernel_size,height*stride);
  	int wstart = max(w * stride - pad,0);
  	int wend = min(w * stride  -pad + kernel_size,width*stride);
  	
  	float hist[21];
  	for(int c=0;c<channels;c++)	
  		hist[c] = 0;
  	float count = 0;
  	for(int ih = hstart;ih<hend;ih++)
  		for(int iw = wstart;iw<wend;iw++)
  		{ 	
  			int cur_ind = (n * stride*height + ih) * stride*width + iw;  						
  			if (in[cur_ind] >= 0 && in[cur_ind] < channels)// never use != , please try < or > instread...
  				hist[int(in[cur_ind])] += (1 - abs(float(ih - (h * stride + 0.5)))/float(stride)) * (1 - abs(float(iw - (w * stride + 0.5)))/float(stride));
  			count += (1 - abs(float(ih - (h * stride + 0.5)))/float(stride)) * (1 - abs(float(iw - (w * stride + 0.5)))/float(stride));;
  		}
  	float max_value = -1;
  	float max_index = -1;
  	for(int c=0;c<channels;c++)
  	{
  		if (hist[c]>max_value)
  		{
  			max_value = hist[c];
    		max_index = c;
  		}
  	}		
  	if(max_value / count > 0.9)
  		out_0[index] = max_index;		
  	else
 			out_0[index] = channels;
 	}		
}    


void ImresizeLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	int num = top[0]->num();
  int height = top[0]->height();
  int width = top[0]->width();
  if (top.size() == 2)
		resize_2_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		 (num*height*width,stride,kernel_size,num_classes,height,width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data(),top[1]->mutable_gpu_data());
	else if (top.size() == 1)
		resize_1_kernel<<<CAFFE_GET_BLOCKS(num*height*width), CAFFE_CUDA_NUM_THREADS>>>
		 (num*height*width,stride,kernel_size,num_classes,height,width,bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
   
  
}


void ImresizeLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{

}

void ImresizeLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	
}

}  // namespace caffe
		
