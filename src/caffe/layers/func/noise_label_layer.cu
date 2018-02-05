
#include <vector>

#include "caffe/layers/func/noise_label_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/solver.hpp"
namespace caffe {

void NoiseLabelLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (bottom[0]->height() == 1 && bottom[0]->width() == 1)
	{
		if (Caffe::gan_type() == "train_dnet")
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
		
			for (int n=0;n<bottom[0]->num();n++)
			{
				//hidden feature
				for (int c=0;c<channels0;c++)
					top[0]->mutable_cpu_data()[n*(channels0+channels1)+c] = bottom[0]->cpu_data()[n*channels0+c];
				//label				
				for (int c=0;c<channels1;c++)
				{	
					top[0]->mutable_cpu_data()[n*(channels0+channels1)+channels0+c] = bottom[1]->cpu_data()[n*channels1+c];
					top[1]->mutable_cpu_data()[n*channels1+c] = bottom[1]->cpu_data()[n*channels1+c];
				}
				int rand_label = caffe_rng_rand()%18;
				top[0]->mutable_cpu_data()[n*(channels0+channels1)+channels0+rand_label] = float(1) - top[0]->cpu_data()[n*(channels0+channels1)+channels0+rand_label];
				//real image have right label
				//fake image have wrong feature
				//fake image does not have label.
			}
		}
		else//g_net
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
		
			for (int n=0;n<bottom[0]->num();n++)
			{
				//hidden feature
				for (int c=0;c<channels0;c++)
				{
					top[0]->mutable_cpu_data()[n*(channels0+channels1)      +c] = bottom[0]->cpu_data()[n*channels0+c];
					top[0]->mutable_cpu_data()[(num+n)*(channels0+channels1)+c] = bottom[0]->cpu_data()[n*channels0+c];
				}
				//label
				for (int c=0;c<channels1;c++)
				{			
					top[0]->mutable_cpu_data()[n*(channels0+channels1)      +channels0+c] = bottom[1]->cpu_data()[n*channels1+c];
					top[0]->mutable_cpu_data()[(num+n)*(channels0+channels1)+channels0+c] = bottom[1]->cpu_data()[n*channels1+c];//feat label
				
					top[1]->mutable_cpu_data()[n*channels1      +c] = bottom[1]->cpu_data()[n*channels1+c];
					top[1]->mutable_cpu_data()[(num+n)*channels1+c] = bottom[1]->cpu_data()[n*channels1+c];//supervised label
				}
				int rand_label = caffe_rng_rand()%18;
				top[0]->mutable_cpu_data()[(num+n)*(channels0+channels1)+channels0+rand_label] = float(1) - top[0]->cpu_data()[(num+n)*(channels0+channels1)+channels0+rand_label];
				top[1]->mutable_cpu_data()[(num+n)*channels1+rand_label] = float(1) - top[1]->cpu_data()[(num+n)*channels1+rand_label];
				//two fake images should have true label and artificial label individually
			}
		}
	}
	else//------------------------------- large feature----------------------------
	{
		int num = bottom[0]->num();
		int channels0 = bottom[0]->channels();
		int channels1 = bottom[1]->channels();
		int height = bottom[0]->height();
		int width = bottom[0]->width();
			
		if (Caffe::gan_type() == "train_dnet")
		{	
			caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
			caffe_copy(bottom[1]->count(),bottom[1]->gpu_data(),top[1]->mutable_gpu_data());
			caffe_copy(bottom[1]->count(),bottom[1]->gpu_data(),top[2]->mutable_gpu_data());
			
			if (Solver::iter()%5 == 0)
				rand_label_ = (Solver::iter()/5)%18;			
			for (int n=0;n<bottom[0]->num();n++)
				top[2]->mutable_cpu_data()[num*channels1+rand_label_] = float(1) - top[2]->cpu_data()[num*channels1+rand_label_];//feat label
			//real image have right label
			//fake image have wrong feature
			//fake image does not have label.
		}
		else//g_net
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
			
			if (Solver::iter()%5 == 0)
				rand_label_ = (Solver::iter()/5)%18;		
			for (int n=0;n<bottom[0]->num();n++)
			{
				//hidden feature
				caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data());
				caffe_copy(bottom[0]->count(),bottom[0]->gpu_data(),top[0]->mutable_gpu_data()+bottom[0]->count());
				//label
				for (int c=0;c<channels1;c++)
				{
					top[1]->mutable_cpu_data()[n*channels1      +c] = bottom[1]->cpu_data()[n*channels1+c];
					top[1]->mutable_cpu_data()[(num+n)*channels1+c] = bottom[1]->cpu_data()[n*channels1+c];//supervised label
					
					top[2]->mutable_cpu_data()[n*channels1      +c] = bottom[1]->cpu_data()[n*channels1+c];
					top[2]->mutable_cpu_data()[(num+n)*channels1+c] = bottom[1]->cpu_data()[n*channels1+c];//feat label
				}	
				top[1]->mutable_cpu_data()[(num+n)*channels1+rand_label_] = float(1) - top[1]->cpu_data()[(num+n)*channels1+rand_label_];
				top[2]->mutable_cpu_data()[(num+n)*channels1+rand_label_] = float(1) - top[2]->cpu_data()[(num+n)*channels1+rand_label_];
				//two fake images should have true label and artificial label individually
			}
		}
	}
}


void NoiseLabelLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	if (bottom[0]->height() == 1 && bottom[0]->width() == 1)
	{
		if (Caffe::gan_type() == "train_dnet")
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			for (int n=0;n<bottom[0]->num();n++)
			{
				//hidden feature
				for (int c=0;c<channels0;c++)
					bottom[0]->mutable_cpu_diff()[n*channels0+c] = top[0]->cpu_diff()[n*(channels0+channels1)+c];
			}
		}
		else
		{
			int num = bottom[0]->num();
			int channels0 = bottom[0]->channels();
			int channels1 = bottom[1]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
		
			for (int n=0;n<bottom[0]->num();n++)
			{
				//hidden feature
				for (int c=0;c<channels0;c++)
				{
					bottom[0]->mutable_cpu_diff()[n*channels0+c] = top[0]->cpu_diff()[n*(channels0+channels1)+c] 
																							+ top[0]->cpu_diff()[(num+n)*(channels0+channels1)+c] ;
				}
			}	
		}
	}
	else//------------------------------large feature-----------------------
	{
		if (Caffe::gan_type() == "train_dnet")
			caffe_copy(bottom[0]->count(),top[0]->gpu_diff(),bottom[0]->mutable_gpu_diff());
		else
		{
			caffe_gpu_add(bottom[0]->count(),float(1),top[0]->gpu_diff(),
																			 float(1),top[0]->gpu_diff()+bottom[0]->count(),
																			 bottom[0]->mutable_gpu_diff());
		}
	}
}


void NoiseLabelLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}


}  // namespace caffe
