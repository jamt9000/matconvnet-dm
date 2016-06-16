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
  opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",             1,   opt_stride              },
  {"Pad",                1,   opt_pad                 },
  {"PoolSwitches",       1,   opt_pool_switches       },
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
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_POOL_SWITCHES, OUT_END
} ;

/* ---------------------------------------------------------------- */
/*                                   pooling_max_switches_dm_kernel */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
pooling_max_switches_dm_kernel
(T* pooled,
 uint8_t* poolSwitches,
 const T* data,
 const int pooledWidth,
 const int pooledHeight,
 const int pooledVolume,
 const int width,
 const int height,
 const int poolWidth,
 const int poolHeight,
 const int strideX,
 const int strideY,
 const int padLeft,
 const int padTop)
{
  int pooledIndex = threadIdx.x + blockIdx.x * blockDim.x;
  if (pooledIndex < pooledVolume) {
    int px = pooledIndex ;
    int py = px / pooledWidth ;
    int pz = py / pooledHeight ;
    px %= pooledWidth ;
    py %= pooledHeight ;
    data += pz * (width*height) ;

    int x1 = px * strideX - padLeft ;
    int y1 = py * strideY - padTop ;
    int x2 = x1 + poolWidth ;
    int y2 = y1 + poolHeight ;
    //int x2 = min(x1 + poolWidth, width) ;
    //int y2 = min(y1 + poolHeight, height) ;
    //x1 = max(x1, 0) ;
    //y1 = max(y1, 0) ;

    T bestValue = (T)(-CUDART_INF_F) ;
    uint8_t switchLocation = 1 ;

    int loc = 1 ;
    int bx=-1;
    int by=-1;
    for (int y = y1 ; y < y2 ; ++y) {
      for (int x = x1 ; x < x2 ; ++x) {
        if(x >= 0 && y >= 0 && x < width && y < height
             && bestValue < data[y * width + x]) {
          bestValue = data[y * width + x] ;
          switchLocation = loc ;
          bx = x; by = y;
        }
        loc += 1 ;
      }
    }

    //if (by*width+bx +pz *(width*height) == 1234) {
    //    printf("index %d data[1234] best %f loc %d\n", pooledIndex, bestValue, (int) switchLocation);
    //}
    pooled[pooledIndex] = bestValue ;
    poolSwitches[pooledIndex] = switchLocation;
  }
}

/* ---------------------------------------------------------------- */
/*                          pooling_max_backward_switches_dm_kernel */
/* ---------------------------------------------------------------- */

template <typename T> __global__ void
pooling_max_backward_switches_dm_kernel
(T* derData,
 const uint8_t* poolSwitches,
 const T* derPooled,
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
    T gradient = 0 ;
    derPooled += z * pooledHeight * pooledWidth;
    poolSwitches += z * pooledHeight * pooledWidth;

    for (int py = py1; py <= py2; ++py) {
      for (int px = px1; px <= px2; ++px) {
        int x1 = px * strideX - padLeft ;
        int y1 = py * strideY - padTop ;

        int loc = poolSwitches[py * pooledWidth + px] - 1 ;
        int lx = loc % poolWidth ;
        int ly = loc / poolWidth ;
        //if (index == 1234) {
        //  printf("index %d loc %d lx %d ly %d x1 %d y1 %d x_data %d y_data %d isloc %d\n", index, loc+1, lx, ly, x1, y1, x_data, y_data, x_data == (x1 + lx) && y_data == (y1 +ly));
        //  printf("py1 %d py2 %d px1 %d px2 %d\n", py1,py2,px1,px2);
        //}
        if(x_data == (x1 + lx) && y_data == (y1 +ly)) {
          gradient += derPooled[py * pooledWidth + px] ;
        }
      }
    }

    derData[index] = gradient;
  }
}

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
  mxArray const *poolSwitchesIn = NULL ;

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

  if (backMode && poolSwitchesIn == NULL) {
    mexErrMsgTxt("Backward requires PoolSwitches") ;
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
  if (data.getHeight() + (padTop+padBottom) < poolHeight ||
      data.getWidth() + (padLeft+padRight) < poolWidth) {
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

  /* Get the output Shape */
  vl::TensorShape outputShape((data.getHeight() + (padTop+padBottom) - poolHeight)/strideY + 1,
                              (data.getWidth()  + (padLeft+padRight) - poolWidth)/strideX + 1,
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

  if (deviceType != vl::GPU) {
    mexErrMsgTxt("Only GPU supported") ;
  }

  if (poolSwitchesIn != NULL) {
    poolSwitches.init(poolSwitchesIn) ;
  }
  
  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
    poolSwitches.initWithZeros(deviceType, vl::vlTypeUInt8, outputShape) ;
  } else {
    derData.initWithZeros(deviceType, dataType, data.getShape()) ;
  }

  // Dispatch
  int height = data.getHeight() ;
  int width = data.getWidth() ;
  int depth = data.getDepth() * data.getSize() ;
  int pooledWidth = (width + (padLeft+padRight) - poolWidth)/strideX + 1 ;
  int pooledHeight = (height + (padTop+padBottom) - poolHeight)/strideY + 1 ;
  int pooledVolume = pooledWidth * pooledHeight * depth ;
  int dataVolume = width * height * depth ;

  if (!backMode) {
    if (dataType == vl::vlTypeFloat) {
      pooling_max_switches_dm_kernel<float>
	<<< vl::divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((float*) output.getMemory(), (uint8_t*) poolSwitches.getMemory(),
	 (float const*) data.getMemory(),
	 pooledHeight, pooledWidth, pooledVolume,
	 height, width,
	 poolHeight, poolWidth,
	 strideY, strideX,
	 padTop, padLeft);
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      pooling_max_switches_dm_kernel<double>
	<<< vl::divideUpwards(pooledVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((double*) output.getMemory(), (uint8_t*) poolSwitches.getMemory(),
	 (double const*) data.getMemory(),
	 pooledHeight, pooledWidth, pooledVolume,
	 height, width,
	 poolHeight, poolWidth,
	 strideY, strideX,
	 padTop, padLeft);
#endif
    }
  } else {
    // Backward
    if (dataType == vl::vlTypeFloat) {
      pooling_max_backward_switches_dm_kernel<float>
	<<< vl::divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((float*) derData.getMemory(), (uint8_t const*) poolSwitches.getMemory(),
	 (float*) derOutput.getMemory(),
	 dataVolume,
	 pooledHeight, pooledWidth,
	 height, width, dataVolume,
	 poolHeight, poolWidth,
	 strideY, strideX,
	 padTop, padLeft);
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      pooling_max_backward_switches_dm_kernel<double>
	<<< vl::divideUpwards(dataVolume, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((double*) derData.getMemory(), (uint8_t const*) poolSwitches.getMemory(),
	 (double*) derOutput.getMemory(),
	 dataVolume,
	 pooledHeight, pooledWidth,
	 height, width, dataVolume,
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
  } else {
    out[OUT_RESULT] = output.relinquish() ;
    if (nout > 1) {
      out[OUT_POOL_SWITCHES] = poolSwitches.relinquish() ;
    }
  }
}
