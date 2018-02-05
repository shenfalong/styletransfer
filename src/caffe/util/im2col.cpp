#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/util/im2col.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {


void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int filter_stride_h, const int filter_stride_w, float* data_col) {
  // effective kernel if we expand the filter_strides (trous)
  const int kernel_h_eff = kernel_h + (kernel_h - 1) * (filter_stride_h - 1);
  const int kernel_w_eff = kernel_w + (kernel_w - 1) * (filter_stride_w - 1);
  int height_col = (height + 2 * pad_h - kernel_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w_eff) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = (c % kernel_w) * filter_stride_w;
    int h_offset = ((c / kernel_w) % kernel_h) * filter_stride_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h - pad_h + h_offset;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}



void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
    const int filter_stride_h, const int filter_stride_w, float* data_im) {
  caffe_set(height * width * channels, float(0), data_im);
  const int patch_h_eff = patch_h + (patch_h - 1) * (filter_stride_h - 1);
  const int patch_w_eff = patch_w + (patch_w - 1) * (filter_stride_w - 1);
  int height_col = (height + 2 * pad_h - patch_h_eff) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w_eff) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = (c % patch_w) * filter_stride_w;
    int h_offset = ((c / patch_w) % patch_h) * filter_stride_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      int h_pad = h * stride_h - pad_h + h_offset;
      for (int w = 0; w < width_col; ++w) {
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
              data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}


}  // namespace caffe
