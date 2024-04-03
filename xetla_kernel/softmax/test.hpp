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

class mat1_4096x2048_bf16_cfg0 {
public:
    static constexpr size_t mat_n = 2048;
    static constexpr size_t mat_m = 4096;
    static constexpr size_t wg_n = mat_n;
    static constexpr size_t wg_m = 2;
    static constexpr size_t sg_n = 2048 / 16;
    static constexpr size_t sg_m = 1;
    using data_type_in = sycl::ext::oneapi::bfloat16;
    using data_type_out = sycl::ext::oneapi::bfloat16;
    using data_type_acc = float;
};
