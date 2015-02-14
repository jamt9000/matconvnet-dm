//
//  datacu.cu
//  matconv
//
//  Created by Andrea Vedaldi on 09/02/2015.
//  Copyright (c) 2015 Andrea Vedaldi. All rights reserved.
//

#ifndef ENABLE_GPU
#error "datacu.cu cannot be compiled without GPU support"
#endif

#include "datacu.hpp"
#include "impl/blashelper.hpp"
#include <cublas_v2.h>

using namespace vl ;

/* -------------------------------------------------------------------
 * CudaHelper
 * ---------------------------------------------------------------- */

vl::CudaHelper::CudaHelper()
: isCublasInitialized(false)
#if ENABLE_CUDNN
, isCudnnInitialized(false), cudnnEnabled(true)
#endif
{ }

vl::CudaHelper::~CudaHelper()
{
  if (isCublasInitialized) {
    cublasDestroy(cublasHandle) ;
    isCublasInitialized = false ;
  }
#ifdef ENABLE_CUDNN
  if (isCudnnInitialized) {
    cudnnDestroy(cudnnHandle) ;
    isCudnnInitialized = false ;
  }
#endif
}

/* -------------------------------------------------------------------
 * getCublasHandle
 * ---------------------------------------------------------------- */

cublasStatus_t vl::CudaHelper::getCublasHandle(cublasHandle_t* handle)
{
  if (!isCublasInitialized) {
    cublasStatus_t stat = cublasCreate(&cublasHandle) ;
    if (stat != CUBLAS_STATUS_SUCCESS) { return stat ; }
    isCublasInitialized = true ;
  }
  *handle = cublasHandle ;
  return CUBLAS_STATUS_SUCCESS ;
}

/* -------------------------------------------------------------------
 * getCudnnHandle
 * ---------------------------------------------------------------- */

#if ENABLE_CUDNN
cudnnStatus_t
vl::CudaHelper::getCudnnHandle(cudnnHandle_t* handle)
{
  if (!isCudnnInitialized) {
    cudnnStatus_t stat = cudnnCreate(&cudnnHandle) ;
    if (stat != CUDNN_STATUS_SUCCESS) { return stat ; }
    isCudnnInitialized = true ;
  }
  *handle = cudnnHandle ;
  return CUDNN_STATUS_SUCCESS ;
}

bool
vl::CudaHelper::getCudnnEnabled() const
{
  return cudnnEnabled ;
}

void
vl::CudaHelper::setCudnnEnabled(bool active)
{
  cudnnEnabled = active ;
}
#endif

/* -------------------------------------------------------------------
 * CuBLAS Errors
 * ---------------------------------------------------------------- */

static const char *
getCublasErrorMessageFromStatus(cublasStatus_t status)
{
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED: return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED: return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE: return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH: return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR: return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED: return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR: return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
    case CUBLAS_STATUS_NOT_SUPPORTED: return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
    case CUBLAS_STATUS_LICENSE_ERROR: return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
  }
  return "CuBLAS unknown status" ;
}

vl::Error
vl::CudaHelper::catchCublasError(cublasStatus_t status, char const * description)
{
  /* if there is no CuBLAS error, do not do anything */
  if (status == CUBLAS_STATUS_SUCCESS) { return vl::vlSuccess ; }

  /* if there is a CuBLAS error, store it */
  lastCublasError = status ;
  std::string message = getCublasErrorMessageFromStatus(status) ;
  if (description) {
    message = std::string(description) + " (" + message + ")" ;
  }
  lastCublasErrorMessage = message ;
  return vl::vlErrorCublas ;
}

cublasStatus_t
vl::CudaHelper::getLastCublasError() const
{
  return lastCublasError;
}

std::string const&
vl::CudaHelper::getLastCublasErrorMessage() const
{
  return lastCublasErrorMessage ;
}

/* -------------------------------------------------------------------
 * CuDNN Errors
 * ---------------------------------------------------------------- */

#if ENABLE_CUDNN
vl::Error
vl::CudaHelper::catchCudnnError(cudnnStatus_t status, char const* description)
{
  /* if there is no CuDNN error, do not do anything */
  if (status == CUDNN_STATUS_SUCCESS) { return vl::vlSuccess ; }

  /* if there is a CuDNN error, store it */
  lastCudnnError = status ;
  std::string message = cudnnGetErrorString(status) ;
  if (description) {
    message = std::string(description) + " (" + message + ")" ;
  }
  lastCudnnErrorMessage = message ;
  return vl::vlErrorCudnn ;
}

cudnnStatus_t
vl::CudaHelper::getLastCudnnError() const
{
  return lastCudnnError;
}

std::string const&
vl::CudaHelper::getLastCudnnErrorMessage() const
{
  return lastCudnnErrorMessage ;
}
#endif

/* -------------------------------------------------------------------
 * Cuda Errors
 * ---------------------------------------------------------------- */

vl::Error
vl::CudaHelper::catchCudaError(char const* description)
{
  /* if there is no Cuda error, do not do anything */
  cudaError_t error = cudaPeekAtLastError() ;
  if (error == cudaSuccess) { return vl::vlSuccess ; }

  /* if there is a Cuda error, eat it and store it */
  lastCudaError = cudaGetLastError() ;
  std::string message = cudaGetErrorString(lastCudaError) ;
  if (description) {
    message = std::string(description) + ": " + message ;
  }
  lastCudaErrorMessage = message ;
  return vl::vlErrorCuda ;
}

cudaError_t
vl::CudaHelper::getLastCudaError() const
{
  return lastCudaError ;
}

std::string const&
vl::CudaHelper::getLastCudaErrorMessage() const
{
  return lastCudaErrorMessage ;
}


