#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>


#include <string>
#include <vector>


#include "caffe/data_transformer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {


DataTransformer::DataTransformer(const TransformationParameter& param): param_(param)
{
  // check if we want to use mean_value
  if (param_.mean_value_size() > 0) 
  {  
    for (int c = 0; c < param_.mean_value_size(); ++c) 
      mean_values_.push_back(param_.mean_value(c));
  }
}


void DataTransformer::Transform(const cv::Mat& cv_img, Blob* transformed_blob)
{
  const int crop_size = param_.crop_size();
  const bool alter_color = param_.alter_color();
  const bool pad_img = param_.pad_img();
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;

  // Check dimensions.
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();


  const bool do_mirror = param_.mirror() && Rand(2);
  
  vector<bool> do_alter_color;
  do_alter_color.resize(3);
  if (alter_color)
  {
  	for(int i=0;i<do_alter_color.size();i++)
    	do_alter_color[i]=true; 
  }
	else
	{
		for(int i=0;i<do_alter_color.size();i++)
    	do_alter_color[i]=false; 
	}
	vector<int> alter_color_value;
  alter_color_value.resize(3);
  alter_color_value[0]=0;
  alter_color_value[1]=0;
  alter_color_value[2]=0;
#if 0
	float a0,a1,a2;
	caffe_rng_gaussian(1, float(0), float(0.1), &a0);
	caffe_rng_gaussian(1, float(0), float(0.1), &a1);
	caffe_rng_gaussian(1, float(0), float(0.1), &a2);
	float range = 1000;
	alter_color_value[0] = a0 * 0.2175 *  0.4009 * range + a1 * 0.0188 * -0.8140 * range + a2 * 0.0045 *  0.4203 * range;
	alter_color_value[1] = a0 * 0.2175 *  0.7192 * range + a1 * 0.0188 * -0.0045 * range + a2 * 0.0045 * -0.6948 * range;
	alter_color_value[2] = a0 * 0.2175 * -0.5675 * range + a1 * 0.0188 * -0.5808 * range + a2 * 0.0045 * -0.5836 * range;
#else  
  for(int i=0;i<alter_color_value.size();i++)
    alter_color_value[i]=Rand(40)-20;
#endif
	
	unsigned int b= mean_values_[0];
	unsigned int g= mean_values_[1];
	unsigned int r= mean_values_[2];

	
	//multi-thread seems to have trouble with opencv's functions
	cv::Mat cv_padded_img = cv_img; 
	if (pad_img)
	{	
		cv::copyMakeBorder(cv_img, cv_padded_img, 
													4, 4, 4, 4, cv::BORDER_CONSTANT,
		                      cv::Scalar(b,g,r) );
		img_height = cv_padded_img.rows;
		img_width = cv_padded_img.cols;
  }                   

  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img;
  if (crop_size)
  {
    CHECK_EQ(crop_size, height);
    CHECK_EQ(crop_size, width);
    // We only do random crop when we do training.
    if (!param_.center_crop())
    {
      h_off = Rand(img_height - crop_size + 1);
      w_off = Rand(img_width - crop_size + 1);
    }
    else
    {
      h_off = (img_height - crop_size) / 2;
      w_off = (img_width - crop_size) / 2;
    }
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_padded_img(roi);
  }
  else
  {
    CHECK_EQ(img_height, height);
    CHECK_EQ(img_width, width);
  }
	//LOG(INFO)<<"w_off = "<<w_off<<"crop_size = "<<crop_size<<" img_height = "<<img_height;
  CHECK(cv_cropped_img.data);

  float* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;

  for (int h = 0; h < height; ++h)
  {
    const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < width; ++w)
      for (int c = 0; c < img_channels; ++c)
      {
        if (do_mirror)
          top_index = (c * height + h) * width + (width - 1 - w);
        else
          top_index = (c * height + h) * width + w;

        float pixel = static_cast<float>(ptr[img_index++]);
        if(do_alter_color[c])
          transformed_data[top_index] = pixel - mean_values_[c] + alter_color_value[c];
        else
          transformed_data[top_index] = pixel - mean_values_[c];
         
      }
  }
    #if 0
    FILE * fid = fopen("debug","wb");
    LOG(INFO)<<transformed_blob->height()<<", width = "<<transformed_blob->width();
    fwrite(transformed_blob->cpu_data(),sizeof(float),32*32*3,fid);
    fclose(fid);
    LOG(FATAL)<<"-------------";
    #endif

}


void DataTransformer::Transformsimple(const cv::Mat& cv_img, Blob* transformed_blob)
{
  const int crop_size = param_.crop_size();
  const bool alter_color = param_.alter_color();
  const bool pad_img = param_.pad_img();
  const int img_channels = cv_img.channels();
  int img_height = cv_img.rows;
  int img_width = cv_img.cols;

  // Check dimensions.
  const int num = transformed_blob->num();
  const int channels = transformed_blob->channels();
  const int height = transformed_blob->height();
  const int width = transformed_blob->width();
	
	if (img_height != height)
		cv::resize(cv_img,cv_img,cv::Size(width,height),0,0,CV_INTER_LINEAR);
    

  float* transformed_data = transformed_blob->mutable_cpu_data();
  int top_index;
  

  for (int h = 0; h < height; ++h)
  {
		const uchar* ptr = cv_img.ptr<uchar>(h);
		int img_index = 0;
		for (int w = 0; w < width; ++w)           
			for (int c = 0; c < img_channels; ++c)
			{   
			  top_index = (c * height + h) * width + w;
			  //pay attention to the format of caffe_rng_rand()!!!
				transformed_data[top_index] = (static_cast<float>(ptr[img_index++]) + float(Rand(1000))/float(1000) - float(127.5))/float(127.5);
				//
			}
  } 
}
 

void DataTransformer::TransformImgAndSeg(const std::vector<cv::Mat>& cv_img_seg,
                                                Blob* transformed_data_blob, Blob* transformed_label_blob, const int ignore_label)
{
  CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";

  const int img_channels = cv_img_seg[0].channels();
  // height and width may change due to pad for cropping
  int img_height   = cv_img_seg[0].rows;
  int img_width    = cv_img_seg[0].cols;

  const int seg_channels = cv_img_seg[1].channels();
  int seg_height   = cv_img_seg[1].rows;
  int seg_width    = cv_img_seg[1].cols;
  
	CHECK_EQ(img_height,seg_height);
	CHECK_EQ(img_width,seg_width);

	

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();


  const int label_channels = transformed_label_blob->channels();
  const int label_height   = transformed_label_blob->height();
  const int label_width    = transformed_label_blob->width();
	
	CHECK_EQ(data_height,label_height);
	CHECK_EQ(data_width, label_width);
	CHECK_EQ(label_channels,1);

  int crop_size = param_.crop_size();
  bool do_mirror =  param_.mirror() && Rand(2);

	
  vector<bool> do_alter_color;
  do_alter_color.resize(3);
  for(int i=0;i<do_alter_color.size();i++)
    do_alter_color[i]=Rand(2);
  vector<int> alter_color_value;
  alter_color_value.resize(3);
  for(int i=0;i<alter_color_value.size();i++)
    alter_color_value[i]=Rand(40)-20;


  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_seg[0];
  cv::Mat cv_cropped_seg = cv_img_seg[1];
  
  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  if(crop_size == 0)
  {
    int pad_height = data_height - img_height;
    int pad_width = data_width - img_width;
    if (pad_height > 0 || pad_width > 0)
    {
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
                         0, pad_width, cv::BORDER_CONSTANT,
                         cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
      cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, 0, pad_height,
                         0, pad_width, cv::BORDER_CONSTANT,
                         cv::Scalar(ignore_label));
      // update height/width
      img_height   = cv_cropped_img.rows;
      img_width    = cv_cropped_img.cols;

      seg_height   = cv_cropped_seg.rows;
      seg_width    = cv_cropped_seg.cols;
    }
  }
  else
  {
    int pad_height = std::max(crop_size - img_height, 0);
    int pad_width  = std::max(crop_size - img_width, 0);
    if (pad_height > 0 || pad_width > 0)
    {
      cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
                         0, pad_width, cv::BORDER_CONSTANT,
                         cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
      cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, 0, pad_height,
                         0, pad_width, cv::BORDER_CONSTANT,
                         cv::Scalar(ignore_label));

      img_height   = cv_cropped_img.rows;
      img_width    = cv_cropped_img.cols;
    }

    h_off = Rand(img_height - crop_size + 1);
    w_off = Rand(img_width - crop_size + 1);
    
    cv::Rect roi(w_off, h_off, crop_size, crop_size);
    cv_cropped_img = cv_cropped_img(roi);
    cv_cropped_seg = cv_cropped_seg(roi);
  }

  
  

  float* transformed_data  = transformed_data_blob->mutable_cpu_data();
  float* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const uchar* label_ptr;


  for (int h = 0; h < data_height; ++h)
  {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_seg.ptr<uchar>(h);
    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w)
    {
      // for image
      for (int c = 0; c < img_channels; ++c)
      {
        if (do_mirror)
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        else
          top_index = (c * data_height + h) * data_width + w;

        float pixel = static_cast<float>(data_ptr[data_index++]);

        if(do_alter_color[c])
          transformed_data[top_index] =pixel - mean_values_[c] + alter_color_value[c];
        else
          transformed_data[top_index] =pixel - mean_values_[c];
      }

      // for segmentation
      if (do_mirror)
        top_index = h * data_width + data_width - 1 - w;
      else
        top_index = h * data_width + w;

      float pixel = static_cast<float>(label_ptr[label_index++]);
      transformed_label[top_index] = pixel;
    }
  }

 
}

void DataTransformer::TransformGan(const std::vector<cv::Mat>& cv_img_seg,
                                                Blob* transformed_data_blob, Blob* transformed_label_blob, const int ignore_label)
{
  CHECK(cv_img_seg.size() == 2) << "Input must contain image and seg.";

  const int img_channels = cv_img_seg[0].channels();
  // height and width may change due to pad for cropping
  int img_height   = cv_img_seg[0].rows;
  int img_width    = cv_img_seg[0].cols;

  const int seg_channels = cv_img_seg[1].channels();
  int seg_height   = cv_img_seg[1].rows;
  int seg_width    = cv_img_seg[1].cols;
  
	CHECK_EQ(img_height,seg_height);
	CHECK_EQ(img_width,seg_width);

	

  const int data_channels = transformed_data_blob->channels();
  const int data_height   = transformed_data_blob->height();
  const int data_width    = transformed_data_blob->width();


  const int label_channels = transformed_label_blob->channels();
  const int label_height   = transformed_label_blob->height();
  const int label_width    = transformed_label_blob->width();
	
	CHECK_EQ(data_height,label_height);
	CHECK_EQ(data_width, label_width);
	CHECK_EQ(label_channels,20);

  int crop_size = param_.crop_size();
  bool do_mirror =  param_.mirror() && Rand(2);

	
  vector<bool> do_alter_color;
  do_alter_color.resize(3);
  for(int i=0;i<do_alter_color.size();i++)
    do_alter_color[i]=Rand(2);
  vector<int> alter_color_value;
  alter_color_value.resize(3);
  for(int i=0;i<alter_color_value.size();i++)
    alter_color_value[i]=Rand(40)-20;


  int h_off = 0;
  int w_off = 0;
  cv::Mat cv_cropped_img = cv_img_seg[0];
  cv::Mat cv_cropped_seg = cv_img_seg[1];
  
  // transform to double, since we will pad mean pixel values
  cv_cropped_img.convertTo(cv_cropped_img, CV_64F);

  // Check if we need to pad img to fit for crop_size
  // copymakeborder
  
  int pad_height = std::max(crop_size - img_height, 0);
  int pad_width  = std::max(crop_size - img_width, 0);
  if (pad_height > 0 || pad_width > 0)
  {
    cv::copyMakeBorder(cv_cropped_img, cv_cropped_img, 0, pad_height,
                       0, pad_width, cv::BORDER_CONSTANT,
                       cv::Scalar(mean_values_[0], mean_values_[1], mean_values_[2]));
    cv::copyMakeBorder(cv_cropped_seg, cv_cropped_seg, 0, pad_height,
                       0, pad_width, cv::BORDER_CONSTANT,
                       cv::Scalar(ignore_label));

    img_height   = cv_cropped_img.rows;
    img_width    = cv_cropped_img.cols;
  }

  h_off = Rand(img_height - crop_size + 1);
  w_off = Rand(img_width - crop_size + 1);
  
  cv::Rect roi(w_off, h_off, crop_size, crop_size);
  cv_cropped_img = cv_cropped_img(roi);
  cv_cropped_seg = cv_cropped_seg(roi);


  
  

  float* transformed_data  = transformed_data_blob->mutable_cpu_data();
  float* transformed_label = transformed_label_blob->mutable_cpu_data();

  int top_index;
  const double* data_ptr;
  const uchar* label_ptr;

	caffe_set(transformed_label_blob->count(),float(0),transformed_label_blob->mutable_cpu_data());
  for (int h = 0; h < data_height; ++h)
  {
    data_ptr = cv_cropped_img.ptr<double>(h);
    label_ptr = cv_cropped_seg.ptr<uchar>(h);
    int data_index = 0;
    int label_index = 0;

    for (int w = 0; w < data_width; ++w)
    {
      // for image
      for (int c = 0; c < img_channels; ++c)
      {
        if (do_mirror)
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        else
          top_index = (c * data_height + h) * data_width + w;

        float pixel = static_cast<float>(data_ptr[data_index++]);

        //if(do_alter_color[c])
        //  transformed_data[top_index] = (pixel - float(127.5) + alter_color_value[c])/float(127.5);
        //else
          transformed_data[top_index] = (pixel - float(127.5) )/float(127.5);
      }

      // for segmentation

      int pixel = static_cast<int>(label_ptr[label_index++]);
      if (pixel == ignore_label)
      	pixel = 19;
      if (do_mirror)
        top_index = (pixel * data_height + h) * data_width + data_width - 1 - w;
      else
        top_index = (pixel * data_height + h) * data_width + w;
      transformed_label[top_index] = float(1);
      
      
      #if 0
      for (int c = 0; c < img_channels; ++c)
      {
        if (do_mirror)
          top_index = (c * data_height + h) * data_width + (data_width - 1 - w);
        else
          top_index = (c * data_height + h) * data_width + w;

        float pixel = static_cast<float>(label_ptr[label_index++]);

        //if(do_alter_color[c])
        //  transformed_data[top_index] = (pixel - float(127.5) + alter_color_value[c])/float(127.5);
        //else
          transformed_label[top_index] = (pixel - float(127.5) )/float(127.5);
      }
      #endif
    }
  }
 
}

int DataTransformer::Rand(int n) {
  CHECK_GT(n, 0);
  return (caffe_rng_rand() % n);
}

}  // namespace caffe
