#pragma once
#include <sycl.hpp>

class mat0_96x2048x2048_bf16 {
public:
    static constexpr size_t mat_n = 2048;
    static constexpr size_t mat_m = 2048 * 96;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;
    static constexpr size_t sg_n = 512;
    static constexpr size_t sg_m = 4;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};

class mat1_96x2048x2048_bf16 {
public:
    static constexpr size_t mat_n = 2048;
    static constexpr size_t mat_m = 2048 * 96;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 1;
    static constexpr size_t sg_n = 2048;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};

class mat1_256x256_bf16_cfg0 {
public:
    static constexpr size_t mat_n = 256;
    static constexpr size_t mat_m = 4096;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;  //1 4 8 16 
    static constexpr size_t sg_n = mat_n;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};

class mat1_1024x1024_bf16_cfg0 {
public:
    static constexpr size_t mat_n = 1024;
    static constexpr size_t mat_m = 4096;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;  //1 4 8 16
    static constexpr size_t sg_n = mat_n;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};

class mat1_2048x2048_bf16_cfg0 {
public:
    static constexpr size_t mat_n = 2048;
    static constexpr size_t mat_m = 4096;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;  //1 4 8 16
    static constexpr size_t sg_n = mat_n;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};

#if 0
class mat1_3072x3072_bf16_cfg0 {
public:
    static constexpr size_t mat_n = 3072;
    static constexpr size_t mat_m = 3072;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;  //1 4 8 16
    static constexpr size_t sg_n = mat_n;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};
#endif
class mat1_4096x4096_bf16_cfg0 {
public:
    static constexpr size_t mat_n = 4096;
    static constexpr size_t mat_m = 4096;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 4;  //1 4 8 16
    static constexpr size_t sg_n = mat_n;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};


