#include "bits/mexutils.h"
#include "bits/datamex.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif
#include <math_constants.h>

#undef printf
#include <stdio.h>

/* option codes */
enum {
  opt_stride = 0,
  opt_pad,
  opt_pool_switches,
  opt_unpool_output_size,
  opt_sum,
  opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",             1,   opt_stride              },
  {"Pad",                1,   opt_pad                 },
  {"PoolSwitches",       1,   opt_pool_switches       },
  {"UnpoolOutputSize",   1,   opt_unpool_output_size  },
  {"Sum",                1,   opt_sum                 },
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
/*                                  unpooling_max_forward_dm_kernel */
/* ---------------------------------------------------------------- */
template <typename T> __global__ void
unpooling_max_forward_dm_kernel
(T* unpooled,
 const T* data,
 const uint8_t* poolSwitches,
 const T* sum,
 const int nthreads,
 const int pooledWidth,
 const int pooledHeight,
 const int width,
 const int height,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - poolWidth ;
    int dy = y_data + padTop - poolHeight ;
    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int px2 = min((x_data + padLeft) / strideX, pooledWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, pooledHeight - 1) ;
    T unpoolValue = (T)(-CUDART_INF_F);
    poolSwitches += z * pooledHeight * pooledWidth ;
    data += z * pooledHeight * pooledWidth ;
    for (int py = py1; py <= py2; ++py) {
      for (int px = px1; px <= px2; ++px) {
        int x1 = px * strideX - padLeft ;
        int y1 = py * strideY - padTop ;

        int loc = poolSwitches[py * pooledWidth + px] - 1 ;
        int lx = loc % poolWidth ;
        int ly = loc / poolWidth ;
        if(x_data == (x1 + lx) && y_data == (y1 + ly)) {
          if (data[py * pooledWidth + px] > unpoolValue) {
            unpoolValue = data[py * pooledWidth + px];
          }
        }
      }
    }
    if (sum) {
        unpoolValue += sum[index] ;
    }
    unpooled[index] = unpoolValue;
  }
}

/* ---------------------------------------------------------------- */
/*                               unpooling_max_backward_dm_kernel   */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
  unpooling_max_backward_dm_kernel
(T* derData,
 T* derSum,
 const T* data,
 const uint8_t* poolSwitches,
 const T* derUnpooled,
 const int nthreads,
 const int pooledWidth,
 const int pooledHeight,
 const int width,
 const int height,
 const int depth,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < nthreads) {
    int x_data = index ;
    int y_data = x_data / width ;
    int z = y_data / height ;
    x_data %= width ;
    y_data %= height ;

    int dx = x_data + padLeft - poolWidth ;
    int dy = y_data + padTop - poolHeight ;
    int px1 = (dx >= 0) ? dx/strideX + 1 : 0 ;
    int py1 = (dy >= 0) ? dy/strideY + 1 : 0 ;
    int px2 = min((x_data + padLeft) / strideX, pooledWidth - 1) ;
    int py2 = min((y_data + padTop) / strideY, pooledHeight - 1) ;
    T unpoolValue = (T)(-CUDART_INF_F);
    T derValue = 0;
    poolSwitches += z * pooledHeight * pooledWidth ;
    derData += z * pooledHeight * pooledWidth ;
    data += z * pooledHeight * pooledWidth ;
    int derDataIndex = -1 ;
    for (int py = py1; py <= py2; ++py) {
      for (int px = px1; px <= px2; ++px) {
	int x1 = px * strideX - padLeft ;
	int y1 = py * strideY - padTop ;

	int loc = poolSwitches[py * pooledWidth + px] - 1 ;
	int lx = loc % poolWidth ;
	int ly = loc / poolWidth ;
	if(x_data == (x1 + lx) && y_data == (y1 + ly)) {
	  if (data[py * pooledWidth + px] > unpoolValue) {
	    unpoolValue = data[py * pooledWidth + px];
	    derDataIndex = py * pooledWidth + px;
	    derValue = derUnpooled[index];
	  }
	}
      }
    }

    if (derDataIndex != -1) {
      derData[derDataIndex] = derValue;
      if (derSum != NULL) {
	derSum[index] = derValue ;
      }
    }
  }
}
/* ---------------------------------------------------------------- */
/*                                                       MEX driver */
/* ---------------------------------------------------------------- */

enum {
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_DERSUM, OUT_END
} ;


void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int poolWidth ;
  int poolHeight ;
  int strideX = 1 ;
  int strideY = 1 ;
  int padLeft = 0 ;
  int padRight = 0 ;
  int padTop = 0 ;
  int padBottom = 0 ;
  bool backMode = false ;
  bool doDerSum = false ;
  mxArray const *poolSwitchesIn = NULL ;
  mxArray const *sumIn = NULL ;
  int unpooledHeight = 0;
  int unpooledWidth = 0;

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

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 3) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            padLeft = (int)mxGetPr(optarg)[0] ;
            padRight = padLeft ;
            padTop = padLeft ;
            padBottom = padLeft ;
            break ;
          case 4:
            padTop = (int)mxGetPr(optarg)[0] ;
            padBottom = (int)mxGetPr(optarg)[1] ;
            padLeft = (int)mxGetPr(optarg)[2] ;
            padRight = (int)mxGetPr(optarg)[3] ;
            break ;
          default:
            mexErrMsgTxt("PAD has neither one nor four elements.") ;
        }
        break;

      case opt_pool_switches :
        poolSwitchesIn = optarg ;
        break ;

      case opt_unpool_output_size :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("UNPOOLOUTPUTSIZE is not a plain matrix.") ;
        }
        if (mxGetNumberOfElements(optarg) >= 2) {
            unpooledHeight = (int)mxGetPr(optarg)[0] ;
            unpooledWidth = (int)mxGetPr(optarg)[1] ;
        } else {
            mexErrMsgTxt("UNPOOLOUTPUTSIZE has less than 2 elements") ;
        }
        break ;

      case opt_sum :
        sumIn = optarg ;
        break ;

      default:
        break ;
    }
  }

  vl::MexTensor data(context) ;
  vl::MexTensor derOutput(context) ;
  vl::MexTensor sum(context) ;

  data.init(in[IN_DATA]) ;
  data.reshape(4) ; // -> 4 dimensions


  if (backMode) {
    derOutput.init(in[IN_DEROUTPUT]) ;
    derOutput.reshape(4) ; // -> 4 dimensions
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
  }

  if (poolSwitchesIn == NULL) {
    mexErrMsgTxt("Unpooling requires PoolSwitches") ;
  }

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
    mexErrMsgTxt("SIZE is not a plain matrix.") ;
  }
  switch (mxGetNumberOfElements(in[IN_SIZE])) {
    case 1:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = poolHeight ;
      break ;
    case 2:
      poolHeight = mxGetPr(in[IN_SIZE])[0] ;
      poolWidth = mxGetPr(in[IN_SIZE])[1] ;
      break ;
    default:
      mexErrMsgTxt("SIZE has neither one nor two elements.") ;
  }

  /* Basic compatibility of Shape */
  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (poolHeight == 0 || poolWidth == 0) {
    mexErrMsgTxt("A dimension of the pooling SIZE is void.") ;
  }
  if (unpooledHeight + (padTop+padBottom) < poolHeight ||
      unpooledWidth + (padLeft+padRight) < poolWidth) {
    mexErrMsgTxt("The pooling window is larger than the DATA (including padding).") ;
  }
  if (padLeft < 0 ||
      padRight < 0 ||
      padTop < 0 ||
      padBottom < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }
  if (padLeft >= poolWidth ||
      padRight >= poolWidth ||
      padTop >= poolHeight  ||
      padBottom >= poolHeight) {
    mexErrMsgTxt("A padding value is larger or equal to the size of the pooling window.") ;
  }


  if (backMode) {
    unpooledHeight = derOutput.getHeight() ;
    unpooledWidth = derOutput.getWidth() ;
  }

  if ((unpooledWidth <= 0 || unpooledHeight <= 0) && !backMode) {
    mexErrMsgTxt("Unpooling requires UnpoolOutputSize") ;
  }

  /* Get the output Shape */
  vl::TensorShape  outputShape(unpooledHeight,
                               unpooledWidth,
                               data.getDepth(),
                               data.getSize()) ;

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::Device deviceType = data.getDeviceType() ;
  vl::Type dataType = data.getDataType() ;
  vl::MexTensor output(context) ;
  vl::MexTensor poolSwitches(context) ;
  vl::MexTensor derData(context) ;
  vl::MexTensor derSum(context) ;

  if (deviceType != vl::GPU) {
    mexErrMsgTxt("Only GPU supported") ;
  }

  if (poolSwitchesIn != NULL) {
    poolSwitches.init(poolSwitchesIn) ;
    if (poolSwitches.getDeviceType() != deviceType) {
      mexErrMsgTxt("PoolSwitches and data have different device type") ;
    }
  }

  if (sumIn != NULL) {
    sum.init(sumIn) ;
    if (! vl::areCompatible(data, sum)) {
      mexErrMsgTxt("DATA and SUM do not have compatible formats.") ;
    }
  }
  
  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
    if (nout > 1) {
      derSum.initWithZeros(deviceType, dataType, derOutput.getShape()) ;
      doDerSum = true ;
    }
  }

  // Dispatch
  int height = outputShape.getHeight() ;
  int width = outputShape.getWidth() ;
  int depth = data.getDepth() * data.getSize() ;
  int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
  int nthreads = width * height * depth ;
  void * sumMem = sumIn ? sum.getMemory() : NULL ;
  void * derSumMem = doDerSum ? derSum.getMemory() : NULL ;

  if (!backMode) {
    if (dataType == vl::vlTypeFloat) {
      unpooling_max_forward_dm_kernel<float>
	<<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((float*) output.getMemory(), (float const*) data.getMemory(),
	 (uint8_t const*) poolSwitches.getMemory(),
         (float const*) sumMem,
	 nthreads,
	 pooledHeight, pooledWidth,
	 height, width, depth,
	 poolHeight, poolWidth,
	 strideY, strideX,
	 padTop, padLeft);
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      unpooling_max_forward_dm_kernel<double>
	<<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((double*) output.getMemory(), (double const*) data.getMemory(),
	 (uint8_t const*) poolSwitches.getMemory(),
         (double const*) sumMem,
	 nthreads,
	 pooledHeight, pooledWidth,
	 height, width, depth,
	 poolHeight, poolWidth,
	 strideY, strideX,
	 padTop, padLeft);
#endif
    }
  } else {
    // Backward
    if (dataType == vl::vlTypeFloat) {
      unpooling_max_backward_dm_kernel<float>
      <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      ((float*) derData.getMemory(), (float*) derSumMem, (float const*) data.getMemory(),
       (uint8_t const*) poolSwitches.getMemory(), (float const*) derOutput.getMemory(),
       nthreads,
       pooledHeight, pooledWidth,
       height, width, depth,
       poolHeight, poolWidth,
       strideY, strideX,
       padTop, padLeft);
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      unpooling_max_backward_dm_kernel<double>
      <<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
      ((double*) derData.getMemory(), (double*) derSumMem, (double const*) data.getMemory(),
       (uint8_t const*) poolSwitches.getMemory(), (double const*) derOutput.getMemory(),
       nthreads,
       pooledHeight, pooledWidth,
       height, width, depth,
       poolHeight, poolWidth,
       strideY, strideX,
       padTop, padLeft);
#endif
    }
  }
   
  cudaError_t status = cudaPeekAtLastError() ;
  if (status != cudaSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str()) ;
  }

  if (backMode) {
    out[OUT_RESULT] = derData.relinquish() ;
    if (doDerSum) {
      out[OUT_DERSUM] = derSum.relinquish() ;
    }
  } else {
    out[OUT_RESULT] = output.relinquish() ;
  }
}
