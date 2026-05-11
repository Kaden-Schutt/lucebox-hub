#pragma once

// Minimal CUDA/HIP runtime compatibility for dflash harness code that already
// uses cuda* names. This is not a HIP-only shim: CUDA builds include the CUDA
// runtime through this header, while HIP builds map the existing cuda* runtime
// spellings to hip*.

#if defined(DFLASH27B_BACKEND_HIP) || defined(GGML_USE_HIP)

#include <hip/hip_runtime.h>

// ── Error / status ──────────────────────────────────────────────────────────
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError

// ── Device management ───────────────────────────────────────────────────────
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaSetDevice hipSetDevice

// ── Memory allocation ───────────────────────────────────────────────────────
#define cudaFree hipFree
#define cudaMalloc hipMalloc

// ── Memcpy / memset ─────────────────────────────────────────────────────────
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemcpy2DAsync hipMemcpy2DAsync
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyKind hipMemcpyKind
#define cudaMemset hipMemset

// ── Streams ─────────────────────────────────────────────────────────────────
#define cudaStream_t hipStream_t
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStreamWaitEvent hipStreamWaitEvent

// ── Events ──────────────────────────────────────────────────────────────────
#define cudaEvent_t hipEvent_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventQuery hipEventQuery

// ── Kernel attribute control ────────────────────────────────────────────────
#define cudaFuncSetAttribute hipFuncSetAttribute

// ── Cross-vendor builtins ───────────────────────────────────────────────────
// HIP has __ballot(predicate) but no masked __ballot_sync(mask, predicate).
// Inference code always uses full-wave mask 0xffffffff, so we drop the mask.
#ifndef __ballot_sync
#define __ballot_sync(mask, predicate) __ballot(predicate)
#endif

#else

#include <cuda_runtime.h>

#endif
