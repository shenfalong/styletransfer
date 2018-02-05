#ifndef CAFFE_RNG_CPP_HPP_
#define CAFFE_RNG_CPP_HPP_

#include <algorithm>
#include <iterator>


#include "caffe/common.hpp"

namespace caffe {


// Fisher–Yates algorithm
template <class RandomAccessIterator>
inline void shuffle(RandomAccessIterator begin, RandomAccessIterator end) 
{
  typedef typename std::iterator_traits<RandomAccessIterator>::difference_type difference_type;


  difference_type length = std::distance(begin, end);
  if (length <= 0) return;
	
	
  for (difference_type i = length - 1; i > 0; --i) {
    std::iter_swap(begin + i, begin + caffe_rng_rand() % (i + 1));
  }
}


}  // namespace caffe

#endif  // CAFFE_RNG_HPP_
