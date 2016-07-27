#include "bits/mexutils.h"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#include <cublas_v2.h>
#endif
#include <math_constants.h>

#undef printf
#include <stdio.h>

/* option codes */
enum {
  opt_stride0 = 0,
  opt_stride1,
  opt_offset0,
  opt_offset1,
  opt_radius,
  opt_alpha,
  opt_sigma,
  opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride0",            1,   opt_stride0             },
  {"Stride1",            1,   opt_stride1             },
  {"Offset0",            1,   opt_offset0             },
  {"Offset1",            1,   opt_offset1             },
  {"Radius",             1,   opt_radius              },
  {"Alpha",              1,   opt_alpha               },
  {"Sigma",              1,   opt_sigma               },
  {"Verbose",            0,   opt_verbose             },
  {0,                    0,   0                       }
} ;

/* ---------------------------------------------------------------- */
/*                                                          Context */
/* ---------------------------------------------------------------- */

vl::MexContext context ;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
 */
void atExit()
{
  context.clear() ;
}

/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_GROUNDTRUTH, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_GRADIENT, OUT_END
} ;

/* ---------------------------------------------------------------- */
/*                           structuredloss_dm_threshold_kernel     */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
structuredloss_dm_threshold_kernel
(T* output,
 const T* data,
 const T* groundTruth,
 const int height,
 const int width,
 const int gridHeight,
 const int gridWidth,
 const int gtHeight,
 const int gtWidth,
 const int stride0,
 const int stride1,
 const int offset0,
 const int offset1,
 const int radius,
 const T alpha,
 const T sigmasq,
 const int nthreads)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int i1 = index % height ;
    int j1 = (index / height) % width;
    int i = index / (height * width) ;
    int i0 = i % gridHeight ;
    int j0 = i / gridHeight ;

    // Find gt pos
    int v0 = stride0 * i0 + offset0 ;
    int u0 = stride0 * j0 + offset0 ;

    float du = groundTruth[v0 + gtHeight * u0] ;
    float dv = groundTruth[v0 + gtHeight * u0 + gtWidth * gtHeight] ;

    float v1 = v0 + dv ;
    float u1 = u0 + du ;

    float ii = (v1-v0)/stride1 + radius ;
    float jj = (u1-u0)/stride1 + radius ;
    
    int gti = (int) round(ii) ;
    int gtj = (int) round(jj) ;

    if (gti == i1 && gtj == j1) {
        output[index] = 0. ;
    }
    else if (gti >= 0 && gti < height && gtj >= 0 && gtj < width) {
      // Ground truth score
      T Sgt = data[gti + height * gtj + width * height * i0
                        + width * height * gridHeight * j0] ;
      if (!isinf(Sgt)) {
        T thresh = alpha + data[index] - Sgt;
        output[index] = (max)((T)0, thresh) ;
      }
    }
  }
}

/* ---------------------------------------------------------------- */
/*                            structuredloss_dm_binarise_kernel     */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
structuredloss_dm_binarise_kernel
(T* output,
 const T* data,
 const T* groundTruth,
 const int height,
 const int width,
 const int gridHeight,
 const int gridWidth,
 const int gtHeight,
 const int gtWidth,
 const int stride0,
 const int stride1,
 const int offset0,
 const int offset1,
 const int radius,
 const T alpha,
 const T sigmasq,
 const int nthreads)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int i1 = index % height ;
    int j1 = (index / height) % width;
    int i = index / (height * width) ;
    int i0 = i % gridHeight ;
    int j0 = i / gridHeight ;

    // Find gt pos
    int v0 = stride0 * i0 + offset0 ;
    int u0 = stride0 * j0 + offset0 ;

    float du = groundTruth[v0 + gtHeight * u0] ;
    float dv = groundTruth[v0 + gtHeight * u0 + gtWidth * gtHeight] ;

    float v1 = v0 + dv ;
    float u1 = u0 + du ;

    float ii = (v1-v0)/stride1 + radius ;
    float jj = (u1-u0)/stride1 + radius ;
    
    int gti = (int) round(ii) ;
    int gtj = (int) round(jj) ;

    if (gti == i1 && gtj == j1) {
        //output[index] = 0. ;
    }
    else if (gti >= 0 && gti < height && gtj >= 0 && gtj < width) {
      // Ground truth score
      T Sgt = data[gti + height * gtj + width * height * i0
                        + width * height * gridHeight * j0] ;
      if (!isinf(Sgt)) {
        T thresh = alpha + data[index] - Sgt;
        if (thresh > 0) {
          output[index] = 0 ;
          atomicAdd(output + gti + height * gtj + width * height * i0
                     + width * height * gridHeight * j0, 1) ;
        }
      }
    }
  }
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int stride0 = 1 ;
  int stride1 = 1 ; 
  int offset0 = 0 ;
  int offset1 = 1 ;
  int radius = 128 ;
  double alpha = 1.0 ;
  double sigma = 0.0 ;
  vl::TensorShape dataSize ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride0 :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE0 is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            stride0 = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE0 has more than one element.") ;
        }
        break ;

      case opt_stride1 :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE0 is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            stride1 = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE1 has more than one element.") ;
        }
        break ;

      case opt_offset0 :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("OFFSET0 is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            offset0 = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("OFFSET0 has more than one element.") ;
        }
        break ;

      case opt_offset1 :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("OFFSET1 is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            offset1 = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("OFFSET1 has more than one element.") ;
        }
        break ;

      case opt_radius :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("RADIUS is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            radius = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("RADIUS has more than one element.") ;
        }
        break ;

      case opt_alpha :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("ALPHA is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            alpha = mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("ALPHA has more than one element.") ;
        }
        break;

      case opt_sigma :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("SIGMA is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            alpha = mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("ALPHA has more than one element.") ;
        }
        break;

      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor groundTruth(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions


  groundTruth.init(in[IN_GROUNDTRUTH]) ;
  groundTruth.reshape(4) ; // -> 4 dimensions

  dataSize = vl::TensorShape(data.getHeight(),
      data.getWidth(),
      data.getDepth(),
      data.getSize()); 

  if (! vl::areCompatible(data, groundTruth)) {
    mexErrMsgTxt("DATA and GROUNDTRUTH do not have compatible formats.") ;
  }


  /* Basic compatibility of Shape */
  if (stride0 < 1 || stride1 < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (offset0 < 0 || offset1 < 0) {
    mexErrMsgTxt("An offset in negative") ;
  }
  if (radius < 1) {
    mexErrMsgTxt("RADIUS is smaller than one.") ;
  }

  /* Create output buffers */
  vl::Device deviceType = data.getDeviceType() ;
  vl::Type dataType = data.getDataType() ;

  vl::MexTensor output(context) ;

  if (deviceType != vl::GPU) {
    mexErrMsgTxt("Only GPU supported") ;
  }
  
  output.initWithZeros(deviceType, dataType, dataSize) ;

  // Dispatch
  int height = dataSize.getHeight() ;
  int width = dataSize.getWidth() ;
  int gridHeight = dataSize.getDepth() ;
  int gridWidth = dataSize.getSize() ;
  int gtHeight = groundTruth.getHeight() ;
  int gtWidth = groundTruth.getWidth() ;
  int nthreads = height * width * gridHeight * gridWidth ;
  float loss  = 0;

  if (dataType == vl::vlTypeFloat) {
    structuredloss_dm_threshold_kernel<float>
      <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      ((float*) output.getMemory(), (const float*) data.getMemory(),
       (const float*) groundTruth.getMemory(),
       height, width, gridHeight, gridWidth, gtHeight, gtWidth, stride0, stride1,
       offset0, offset1, radius, alpha, sigma * sigma, nthreads);

    cublasHandle_t handle ;
    cublasStatus_t status ;
    status = context.getCudaHelper().getCublasHandle(&handle) ;

    if (status != CUBLAS_STATUS_SUCCESS) {
      mexErrMsgTxt("Previous CUBLAS Error in vl_nnstructuredlossdm") ;
    }

    status = cublasSasum(handle, nthreads, (const float*) output.getMemory(), 1, &loss) ;

    if (status != CUBLAS_STATUS_SUCCESS) {
      mexErrMsgTxt("CUBLAS Error in vl_nnstructuredlossdm") ;
    }

    structuredloss_dm_binarise_kernel<float>
      <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      ((float*) output.getMemory(), (const float*) data.getMemory(),
       (const float*) groundTruth.getMemory(),
       height, width, gridHeight, gridWidth, gtHeight, gtWidth, stride0, stride1,
       offset0, offset1, radius, alpha, sigma * sigma, nthreads);

  } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
  // todo
  mexErrMsgTxt("DOUBLE precision not implemented") ;
#endif
  }
  
   
  cudaError_t status = cudaPeekAtLastError() ;
  if (status != cudaSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }

  out[OUT_RESULT] = mxCreateDoubleScalar(loss);

  if (nout > 1) {
    out[OUT_GRADIENT] = output.relinquish() ;
  }
}
