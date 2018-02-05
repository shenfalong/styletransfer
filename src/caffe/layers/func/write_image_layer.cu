#include "caffe/solver.hpp"
#include <vector>
#include "caffe/util/format.hpp"

#include "caffe/layers/func/write_image_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#define IMAGE_NUM 2
namespace caffe {



void WriteImageLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
	if (bottom.size() == 1)
	{
		if (Solver::iter() % 1000 == 0)
	 	{
			LOG(INFO)<<"---------------writing image-----------------";
			std::vector<float> mean_values_;
			mean_values_.clear();
			mean_values_.resize(3);
			mean_values_[0] = 104.008;
		  mean_values_[1] = 116.669;
		  mean_values_[2] = 122.675;
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
			const float * bottom_data = bottom[0]->cpu_data();
			cv::Mat cv_im(height*1,width*1,CV_8UC3);
			//cv::Mat cv_im(height*8,width*8,CV_8UC1);
			for (int i=0;i<1*height;i++)
			{
				unsigned char * data_ptr = cv_im.ptr<uchar>(i);
				for (int j=0;j<1*width;j++)						
				{
					for (int c=0;c<channels;c++)
					{				
						int n = (i/height)*1+(j/width);
						int h = i%height;
						int w = j%width;
						int index = ((n*channels+c)*height+h)*width+w;
						data_ptr[j*channels+c] = min(max(bottom_data[index] + mean_values_[c],float(0)),float(255));		
					}
				}
			}
			std::stringstream ss;
			string filename;
			int gpu_id_;
			CUDA_CHECK(cudaGetDevice(&gpu_id_));
			ss<<"generateimage//"<<Solver::iter()<<"GPU"<<gpu_id_<<".jpg";
			ss>>filename;
			cv::imwrite(filename,cv_im);
		}
	}
	else if (bottom.size() == 2)
	{
		if (Solver::iter()%500 == 0 &&  Caffe::gan_type() == "train_gnet")//
 		{
{	
			LOG(INFO)<<"---------------writing image-----------------";
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
			const float * bottom_data = bottom[1]->cpu_data();
			cv::Mat cv_im(height*IMAGE_NUM ,width*IMAGE_NUM ,CV_8UC3);
			//cv::Mat cv_im(height*IMAGE_NUM,width*IMAGE_NUM,CV_8UC1);
			for (int i=0;i<IMAGE_NUM *height;i++)
			{
				unsigned char * data_ptr = cv_im.ptr<uchar>(i);
				for (int j=0;j<IMAGE_NUM *width;j++)						
				{
					for (int c=0;c<channels;c++)
					{				
						int n = (i/height)*IMAGE_NUM +(j/width);
						int h = i%height;
						int w = j%width;
						int index = ((n*channels+c)*height+h)*width+w;
						data_ptr[j*channels+c] = min(max((bottom_data[index]*127.5+127.5),float(0)),float(255));
					}
				}
			}
			std::stringstream ss;
			string filename;
			ss<<"generateimage//"<<Solver::iter()<<"_"<<format_int(gpu_id_)<<"_real_.jpg";
			ss>>filename;
			cv::imwrite(filename,cv_im);
}
{		
			LOG(INFO)<<"---------------writing image-----------------";
			int num = bottom[0]->num();
			int channels = bottom[0]->channels();
			int height = bottom[0]->height();
			int width = bottom[0]->width();
			const float * bottom_data = bottom[0]->cpu_data();
			cv::Mat cv_im(height*IMAGE_NUM ,width*IMAGE_NUM ,CV_8UC3);
			//cv::Mat cv_im(height*IMAGE_NUM,width*IMAGE_NUM,CV_8UC1);
			for (int i=0;i<IMAGE_NUM *height;i++)
			{
				unsigned char * data_ptr = cv_im.ptr<uchar>(i);
				for (int j=0;j<IMAGE_NUM *width;j++)						
				{
					for (int c=0;c<channels;c++)
					{				
						int n = (i/height)*IMAGE_NUM +(j/width);
						int h = i%height;
						int w = j%width;
						int index = ((n*channels+c)*height+h)*width+w;
						data_ptr[j*channels+c] = min(max((bottom_data[index]*127.5+127.5),float(0)),float(255));
					}
				}
			}
			std::stringstream ss;
			string filename;
			ss<<"generateimage//"<<Solver::iter()<<"_"<<format_int(gpu_id_)<<"_.jpg";
			ss>>filename;
			cv::imwrite(filename,cv_im);
}
		}
	}
}


void WriteImageLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	
}

void WriteImageLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
}

}  // namespace caffe
