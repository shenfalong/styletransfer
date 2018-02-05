#include <vector>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include "caffe/layers/activation/image_resize_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {




void ImageResizeLayer::Forward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{
		int num = bottom[0]->num();
	int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  
  cv::Mat cv_interp_im(height/interp_ratio,width/interp_ratio,CV_32FC3);
	cv::Mat cv_im(height,width,CV_32FC3);
	
	float * top_data = top[0]->mutable_cpu_data();
	for (int n=0;n<num;n++)
	{
		const float * bottom_data = bottom[0]->cpu_data() + bottom[0]->offset(n);
		for (int h = 0;h < height; h++)
	  {
    	uchar * data_ptr = cv_im.ptr(h);
    	for (int w = 0;w < width; w++)
    		for(int c = 0;c < 3; c++)
      		data_ptr[w*3+c] = bottom_data[((c*height)+h)*width+w];
    }		
    

		cv::resize(cv_im,cv_interp_im,cv::Size(width/interp_ratio,height/interp_ratio),0,0,CV_INTER_AREA);
		
	  float * top_data = top[0]->mutable_cpu_data() + top[0]->offset(n);
		for (int h = 0; h < height/interp_ratio; h++)
	  {
    	const uchar * data_ptr = cv_interp_im.ptr(h);
    	for (int w = 0;w < width/interp_ratio; w++)
    		for(int c = 0;c < 3; c++)
      		top_data[int(((c*height/interp_ratio)+h)*width/interp_ratio+w)] = data_ptr[w*3+c];
    }		
  }
	

}


void ImageResizeLayer::Backward_gpu(const vector<Blob*>& top, const vector<Blob*>& bottom) 
{
	
}

void ImageResizeLayer::SecForward_gpu(const vector<Blob*>& bottom, const vector<Blob*>& top) 
{

}


}  // namespace caffe
		
