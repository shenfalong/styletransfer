// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "caffe/util/benchmark.hpp"
#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using cv::Mat;
using cv::Vec3b;
using std::fstream;
using std::ios;
using std::max;
using std::string;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace caffe {


void ReadImageToProto(const string& filename, BlobProto* proto) {
  Mat cv_img;
  cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  CHECK(cv_img.data) << "Could not open or find the image.";
  DCHECK_EQ(cv_img.channels(), 3);
  proto->set_num(1);
  proto->set_channels(3);
  proto->set_height(cv_img.rows);
  proto->set_width(cv_img.cols);
  proto->clear_data();
  proto->clear_diff();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        proto->add_data(static_cast<float>(cv_img.at<Vec3b>(h, w)[c]) / 255.);
      }
    }
  }
}

bool matchExt(const std::string & fn, std::string en) 
{
  size_t p = fn.rfind('.');
  std::string ext = p != fn.npos ? fn.substr(p) : fn;
  std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
  std::transform(en.begin(), en.end(), en.begin(), ::tolower);
  if ( ext == en )
    return true;
  else if ( en == ".jpg" && ext == ".jpeg" )
    return true;  
	else
  	return false;
}

void ReadImageToDatum(const string& filename, const bool is_color, const int label, const std::string & encoding, Datum* datum) 
{
	cv::Mat cv_img;
	if (is_color)
		 cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
	else
		 cv_img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		
	if (!cv_img.data) LOG(FATAL)<<"image "<<filename<<" not found.";
		

#if 0
  
	int width = cv_img.cols;
	int height = cv_img.rows;
  if(width < height)
  {
    height = 480 / float(width) * float(height);
    width = 480;
    cv::resize(cv_img,cv_img,cv::Size(width,height),0,0,CV_INTER_CUBIC);
  }
  else
  {
    width = 480 / float(height) * float(width);
    height = 480;
    cv::resize(cv_img,cv_img,cv::Size(width,height),0,0,CV_INTER_CUBIC);
  }
   
	std::vector<uchar> buf;
	cv::imencode("."+encoding, cv_img, buf);
	datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]), buf.size()));
	datum->set_label(label);
	datum->set_encoded(true);
#else
	std::streampos size;	
	fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
	if (file.is_open()) 
	{
	  size = file.tellg();
	  std::string buffer(size, ' ');
	  file.seekg(0, ios::beg);
	  file.read(&buffer[0], size);
	  file.close();
	  datum->set_data(buffer);
	  datum->set_label(label);
	  datum->set_encoded(true);
	} 
#endif
}
void ReadImageToDatumMultilabel(const string& filename, const bool is_color, const vector<int> label, const std::string & encoding, Datum* datum) 
{
	cv::Mat cv_img;
	if (is_color)
		 cv_img = cv::imread(filename+".jpg", CV_LOAD_IMAGE_COLOR);
	else
		 cv_img = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
		
	if (!cv_img.data) LOG(FATAL)<<"image "<<filename<<" not found.";
		

	std::streampos size;	
	fstream file(filename.c_str(), ios::in|ios::binary|ios::ate);
	if (file.is_open()) 
	{
	  size = file.tellg();
	  std::string buffer(size, ' ');
	  file.seekg(0, ios::beg);
	  file.read(&buffer[0], size);
	  file.close();
	  datum->set_data(buffer);
	  for (int i=0;i<label.size();i++)
	  	datum->add_multi_label(label[i]);
	  datum->set_encoded(true);
	} 
}

void WriteProtoToImage(const string& filename, const BlobProto& proto) {
  CHECK_EQ(proto.num(), 1);
  CHECK(proto.channels() == 3 || proto.channels() == 1);
  CHECK_GT(proto.height(), 0);
  CHECK_GT(proto.width(), 0);
  Mat cv_img(proto.height(), proto.width(), CV_8UC3);
  if (proto.channels() == 1) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          cv_img.at<Vec3b>(h, w)[c] =
              uint8_t(proto.data(h * cv_img.cols + w) * 255.);
        }
      }
    }
  } else {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < cv_img.rows; ++h) {
        for (int w = 0; w < cv_img.cols; ++w) {
          cv_img.at<Vec3b>(h, w)[c] =
              uint8_t(proto.data((c * cv_img.rows + h) * cv_img.cols + w)
                  * 255.);
        }
      }
    }
  }
  CHECK(cv::imwrite(filename, cv_img));
}

void ReadProtoFromTextFile(const char* filename,
    ::google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
  	LOG(FATAL)<<"can not find file "<<filename;
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}

cv::Mat DecodeDatumToCVMat(const Datum& datum, bool is_color) {
  cv::Mat cv_img;
  CHECK(datum.encoded()) << "Datum not encoded";
  const string& data = datum.data();
  std::vector<char> vec_data(data.c_str(), data.c_str() + data.size());
  int cv_read_flag = (is_color ? CV_LOAD_IMAGE_COLOR :
    CV_LOAD_IMAGE_GRAYSCALE);
  cv_img = cv::imdecode(vec_data, cv_read_flag);
  if (!cv_img.data) {
    LOG(ERROR) << "Could not decode datum ";
  }
  return cv_img;
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  if (fd < 0)
  	LOG(FATAL)<<"can not find file "<<filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
	coded_input->SetTotalBytesLimit(INT_MAX, 536870912);

  CHECK(proto->ParseFromCodedStream(coded_input));

  delete coded_input;
  delete raw_input;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

}  // namespace caffe
