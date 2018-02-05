#include "caffe/filler.hpp"
namespace caffe {

Filler* GetFiller(const FillerParameter& param) {
  const std::string& type = param.type(); 
  if (type == "binary") 
    return new BinaryFiller(param);
  else if (type == "msra") 
    return new MSRAFiller(param);
  else if (type == "bilinear") 
    return new BilinearFiller(param);
  else if (type == "potts") 
    return new PottsFiller(param);  
  else if (type == "gaussian") 
    return new GaussianFiller(param);
  else if (type == "glorot") 
    return new GlorotFiller(param);
  else 
    CHECK(false) << "Unknown filler name: " << param.type();
  return (Filler*)(NULL);
}
}
