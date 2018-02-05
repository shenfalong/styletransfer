// The main caffe test code. Your test cpp code should include this hpp
// to allow a main function to be compiled into the binary.

#include "caffe/caffe.hpp"
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <cstdlib>

#include "caffe/common.hpp"

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  caffe::GlobalInit(&argc, &argv);


  std::cout<<"------------------------------------------------------------------"<<'\n';
  std::cout<<"-----------------------test on GPU--------------------------------"<<'\n';
  std::cout<<"------------------------------------------------------------------"<<'\n';
  caffe::Caffe::GPUs.clear();
  caffe::Caffe::GPUs.resize(2);
  caffe::Caffe::GPUs[0]=0;
  caffe::Caffe::GPUs[1]=1;
  // it will make a instance of Caffe at the first time

  // invoke the test.
  return RUN_ALL_TESTS();
}
