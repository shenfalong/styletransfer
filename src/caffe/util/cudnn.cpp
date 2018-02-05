#include "caffe/util/cudnn.hpp"


namespace caffe {

namespace cudnn {

float dataType::oneval = 1.0;
float dataType::zeroval = 0.0;
const void* dataType::one  = static_cast<void *>(&oneval);
const void* dataType::zero = static_cast<void *>(&zeroval);

}  // namespace cudnn

}  // namespace caffe
