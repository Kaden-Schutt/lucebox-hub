#pragma once

// Compatibility include for vendored ggml sources and dflash HIP code that
// references <cuda_bf16.h>. On AMD HIP builds this must resolve to
// hip_bf16 instead of the CUDA SDK, with the CUDA-spelling bf16 type
// aliased to its HIP equivalent.

#include <hip/hip_bf16.h>

#ifndef __nv_bfloat16
using __nv_bfloat16 = __hip_bfloat16;
#endif

#ifndef __nv_bfloat162
using __nv_bfloat162 = __hip_bfloat162;
#endif
