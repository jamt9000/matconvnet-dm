// @file vl_nnaggregate.cu
// @brief DM Aggregation
// @author Andrea Vedaldi

/*
Copyright (C) 2014-15 Andrea Vedaldi
All rights reserved.

This file is part of the VLFeat library and is made available under
the terms of the BSD license (see the COPYING file).
*/

#include "bits/mexutils.h"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

template<typename T> __global__ void
aggregate_kernel
(T* out,
 const T* data,
 const int height,
 const int width,
 const int nthreads,
 const int inGridHeight,
 const int inGridWidth,
 const int outGridHeight,
 const int outGridWidth,
 const int delta,
 const int pad)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < nthreads) {
    int h = index % height ;
    int w = (index / height) % width ;
    int gh = (index / height / width) % outGridHeight ;
    int gw = (index / height / width / outGridHeight) % outGridWidth ;

    int summed = 0 ;
    for (int dw = 0; dw <= 1; ++dw) {
      for (int dh = 0; dh <= 1; ++dh) {
        int gh_in = gh + delta * dh - pad ;
        int gw_in = gw + delta * dw - pad ;
        if (gh_in >= 0 && gh_in < inGridHeight && gw_in >= 0 && gw_in < inGridWidth) {

          summed++ ;
          out[index] += data[h + width * w + width * height * gh_in
                             + width * height * inGridHeight * gw_in] ;
        }
      }
    }
  out[index] /= summed ;
  }
}

template<typename T> __global__ void
aggregate_kernel_back
(T* out,
 const T* derOutput,
 const int height,
 const int width,
 const int nthreads,
 const int inGridHeight,
 const int inGridWidth,
 const int outGridHeight,
 const int outGridWidth,
 const int delta,
 const int pad)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;

  if (index < nthreads) {
    int h = index % height ;
    int w = (index / height) % width ;
    int gh_in = (index / height / width) % inGridHeight ;
    int gw_in = (index / height / width / inGridHeight) % inGridWidth ;

    for (int dw = 0; dw <= 1; ++dw) {
      for (int dh = 0; dh <= 1; ++dh) {
        int gh_out = gh_in - delta * dh + pad ;
        int gw_out = gw_in - delta * dw + pad ;
        if (gh_out >= 0 && gh_out < outGridHeight && gw_out >= 0 && gw_out < outGridWidth) {
          int summed = ((gh_out + delta - pad) >= 0 && (gh_out + delta - pad) < inGridHeight
                          && (gw_out + delta - pad) >= 0 && (gw_out + delta - pad) < inGridWidth)
                        + ((gh_out - pad) >= 0 && (gh_out - pad) < inGridHeight
                          && (gw_out - pad) >= 0 && (gw_out - pad) < inGridWidth)
                        + ((gh_out - pad) >= 0 && (gh_out - pad) < inGridHeight
                          && (gw_out + delta - pad) >= 0 && (gw_out + delta - pad) < inGridWidth)
                        + ((gh_out + delta - pad) >= 0 && (gh_out + delta - pad) < inGridHeight
                          && (gw_out - pad) >= 0 && (gw_out - pad) < inGridWidth);

          out[index] += derOutput[h + width * w + width * height * gh_out
                                  + width * height * outGridHeight * gw_out]/summed;
        }
      }
    }
  }
}

/* option codes */
enum {
  opt_pad = 0,
} ;

/* options */
vlmxOption  options [] = {
  {"Pad",                1,   opt_pad                 },
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
  IN_DATA = 0, IN_DELTA, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  int pad = 0 ;
  int delta ;

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

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            pad = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("PAD has > one element.") ;
        }
        break;

      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions


  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_DELTA],-1,-1)) {
    mexErrMsgTxt("DELTA is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_DELTA])) {
    case 1:
      delta = mxGetPr(in[IN_DELTA])[0] ;
      break ;
    default:
      mexErrMsgTxt("DELTA has > one element.") ;
  }

  /* Get the output Shape */
  vl::TensorShape outputShape(data.getHeight(),
                              data.getWidth(),
                              (data.getDepth()/2) * 2 + 1,
                              (data.getSize()/2) * 2 + 1);



  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::Device deviceType = data.getDeviceType() ;
  vl::Type dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (deviceType != vl::GPU) {
    mexErrMsgTxt("Only GPU supported") ;
  }
  
  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::Error error ;
  if (!backMode) {
    const int nthreads = output.getHeight() * output.getWidth() * output.getDepth() * output.getSize() ;
    if (dataType == vl::vlTypeFloat) {
      aggregate_kernel<float> <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        ((float*)output.getMemory(), (float*)data.getMemory(), data.getHeight(), data.getWidth(), nthreads,
         data.getDepth(), data.getSize(), output.getDepth(), output.getSize(), delta, pad);
      error = vl::vlSuccess ;
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      aggregate_kernel<double> <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        ((double*)output.getMemory(), (double*)data.getMemory(), data.getHeight(), data.getWidth(), nthreads,
         data.getDepth(), data.getSize(), output.getDepth(), output.getSize(), delta, pad);
      error = vl::vlSuccess ;
#endif
    }

  } else {
    const int nthreads = derData.getHeight() * derData.getWidth() * derData.getDepth() * derData.getSize() ;
    if (dataType == vl::vlTypeFloat) {
      aggregate_kernel_back<float> <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        ((float*)derData.getMemory(), (float*)derOutput.getMemory(), data.getHeight(), data.getWidth(), nthreads,
         data.getDepth(), data.getSize(), derOutput.getDepth(), derOutput.getSize(), delta, pad);
      error = vl::vlSuccess ;
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      aggregate_kernel_back<double> <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
        ((double*)derData.getMemory(), (double*)derOutput.getMemory(), data.getHeight(), data.getWidth(), nthreads,
         data.getDepth(), data.getSize(), derOutput.getDepth(), derOutput.getSize(), delta, pad);
      error = vl::vlSuccess ;
#endif
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::vlSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
