//
// Created by chengjun on 2/20/24.
//

#ifndef TRITONBENCHMARK_SOFTMAX_H
#define TRITONBENCHMARK_SOFTMAX_H

#include "test.hpp"

template<typename Config>
void softmax_forward(void* input, void* output, sycl::queue& queue);

#endif //TRITONBENCHMARK_SOFTMAX_H
