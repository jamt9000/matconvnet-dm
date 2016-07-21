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
  opt_stride0 = 0,
  opt_stride1,
  opt_offset0,
  opt_offset1,
  opt_radius,
  opt_pad,
  opt_rectifyborder,
  opt_data_size,
  opt_verbose,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride0",            1,   opt_stride0             },
  {"Stride1",            1,   opt_stride1             },
  {"Offset0",            1,   opt_offset0             },
  {"Offset1",            1,   opt_offset1             },
  {"Radius",             1,   opt_radius              },
  {"Pad",                1,   opt_pad                 },
  {"RectifyBorder",      0,   opt_rectifyborder       },
  {"DataSize",           1,   opt_data_size           },
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
  IN_DATA = 0, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

/* ---------------------------------------------------------------- */
/*                                   cropradius_dm_kernel           */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
cropradius_dm_kernel
(T* output,
 const T* data,
 const int height,
 const int width,
 const int gridHeight,
 const int gridWidth,
 const int stride0,
 const int stride1,
 const int offset0,
 const int offset1,
 const int pad,
 const int radius,
 const int nthreads,
 const bool rectifyBorder)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int cropsize = 2 *radius + 1 ;
    int ri = index % cropsize ;
    int rj = (index / cropsize) % cropsize ;
    int i = index / (cropsize * cropsize) ;
    int i0 = i % gridHeight ;
    int j0 = i / gridHeight ;
    int i0_pixels = i0 * stride0 + offset0 ;
    int j0_pixels = j0 * stride0 + offset0 ;
    int i1 = (i0_pixels - offset1)/stride1 + (ri - radius) + pad ;
    int j1 = (j0_pixels - offset1)/stride1 + (rj - radius) + pad ;

    //if (i0 == 0 &&j0 == 0 && ri == 0) {
    //    printf("i0 %d j0 %d rj %d j1 %d\n", i0, j0, rj, j1) ;
    //}

    if (i1 >= 0 && i1 < height && j1 >= 0 && j1 < width) {
      T out = data[i1 + height * j1 + width * height * i0
                        + width * height * gridHeight * j0] ;
      T nMissing ;
      if (rectifyBorder) {
        // TODO don't hardcode 4
        if (i1 < pad) {
          nMissing = 4 - pad + i1 ;
          out *= 4. / nMissing ;
        }
        if (j1 < pad) {
          nMissing = 4 - pad + j1 ;
          out *= 4. / nMissing ;
        }
        if (height - i1 - 1 < pad) {
          nMissing = 4 - pad + (height - i1 - 1) ; 
          out *= 4. / nMissing ;
        }
        if (width - j1 - 1 < pad) {
          nMissing = 4. - pad + (width - j1 - 1) ;
          out *= 4. / nMissing ;
        }
      }

      output[index] = out ;
    }
  }
}


/* ---------------------------------------------------------------- */
/*                                 cropradius_backward_dm_kernel    */
/* ---------------------------------------------------------------- */

template<typename T> __global__ void
cropradius_backward_dm_kernel
(T* derData,
 const T* derOutput,
 const int height,
 const int width,
 const int gridHeight,
 const int gridWidth,
 const int stride0,
 const int stride1,
 const int offset0,
 const int offset1,
 const int pad,
 const int radius,
 const int nthreads,
 const bool rectifyBorder)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < nthreads) {
    int cropsize = 2 *radius + 1 ;
    int ri = index % cropsize ;
    int rj = (index / cropsize) % cropsize ;
    int i = index / (cropsize * cropsize) ;
    int i0 = i % gridHeight ;
    int j0 = i / gridHeight ;
    int i0_pixels = i0 * stride0 + offset0 ;
    int j0_pixels = j0 * stride0 + offset0 ;
    int i1 = (i0_pixels - offset1)/stride1 + (ri - radius) + pad ;
    int j1 = (j0_pixels - offset1)/stride1 + (rj - radius) + pad ;

    //if (i0 == 0 &&j0 == 0 && ri == 0) {
    //    printf("i0 %d j0 %d rj %d j1 %d\n", i0, j0, rj, j1) ;
    //}

    if (i1 >= 0 && i1 < height && j1 >= 0 && j1 < width) {
      T out = derOutput[index] ;

      T nMissing ;
      if (rectifyBorder) {
        // TODO don't hardcode 4
        if (i1 < pad) {
          nMissing = 4 - pad + i1 ;
          out *= 4. / nMissing ;
        }
        if (j1 < pad) {
          nMissing = 4 - pad + j1 ;
          out *= 4. / nMissing ;
        }
        if (height - i1 - 1 < pad) {
          nMissing = 4 - pad + (height - i1 - 1) ; 
          out *= 4. / nMissing ;
        }
        if (width - j1 - 1 < pad) {
          nMissing = 4. - pad + (width - j1 - 1) ;
          out *= 4. / nMissing ;
        }
      }

      derData[i1 + height * j1 + width * height * i0
            + width * height * gridHeight * j0] = out ;
    }
  }
}

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;
  int stride0 = 1 ;
  int stride1 = 1 ; 
  int offset0 = 0 ;
  int offset1 = 1 ;
  int radius = 128 ;
  int pad = 0 ;
  bool rectifyBorder = false;
  vl::TensorShape dataSize ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit) ;

  if (nin < 1) {
    mexErrMsgTxt("The arguments are less than one.") ;
  }

  if (nin > 1 && vlmxIsString(in[1],-1)) {
    next = 1 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 2) ;
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

      case opt_pad :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("PAD is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            pad = (int)mxGetPr(optarg)[0] ;
            break ;
          default:
            mexErrMsgTxt("PAD has more than one element.") ;
        }
        break;

      case opt_rectifyborder:
        rectifyBorder = true ;
        break ;

      case opt_data_size :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("DATASIZE is not a plain matrix.") ;
        }
        if (mxGetNumberOfElements(optarg) == 4) {
          dataSize = vl::TensorShape((int)mxGetPr(optarg)[0],
                                     (int)mxGetPr(optarg)[1],
                                     (int)mxGetPr(optarg)[2],
                                     (int)mxGetPr(optarg)[3]);
        } else {
          mexErrMsgTxt("DATASIZE does not have 4 elements.") ;
        }
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
    if (!dataSize.isEmpty() && !data.isEmpty()) {
      dataSize = vl::TensorShape(data.getHeight(),
          data.getWidth(),
          data.getDepth(),
          data.getSize());
    }
  } else {
    dataSize = vl::TensorShape(data.getHeight(),
        data.getWidth(),
        data.getDepth(),
        data.getSize());
  }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    if (!data.isEmpty()) {
        mexErrMsgTxt("DATA and DEROUTPUT do not have compatible formats.") ;
    }
  }


  /* Basic compatibility of Shape */
  if (stride0 < 1 || stride1 < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.") ;
  }
  if (pad < 0) {
    mexErrMsgTxt("An element of PAD is negative.") ;
  }
  if (offset0 < 0 || offset1 < 0) {
    mexErrMsgTxt("An offset in negative") ;
  }
  if (radius < 1) {
    mexErrMsgTxt("RADIUS is smaller than one.") ;
  }

  /* Get the output Shape */
  vl::TensorShape outputShape(2 * radius + 1,
                              2 * radius + 1,
                              dataSize.getDepth(),
                              dataSize.getSize()) ;

  if (backMode && (derOutput != outputShape)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.") ;
  }

  /* Create output buffers */
  vl::Device deviceType ;
  vl::Type dataType;
  if (backMode) {
    // data can be CPU since its memory is not used
    deviceType = derOutput.getDeviceType() ;
    dataType = derOutput.getDataType() ;
  } else {
    deviceType = data.getDeviceType() ;
    dataType = data.getDataType() ;
  }
  vl::MexTensor output(context) ;
  vl::MexTensor derData(context) ;

  if (deviceType != vl::GPU) {
    mexErrMsgTxt("Only GPU supported") ;
  }

  
  if (!backMode) {
    output.initWithZeros(deviceType, dataType, outputShape) ;
  } else {
    if (dataSize.isEmpty()) {
      mexErrMsgTxt("Must provide DATASIZE for backward pass") ;
    }
    derData.initWithZeros(deviceType, dataType, dataSize) ;
  }

  // Dispatch
  int height = dataSize.getHeight() ;
  int width = dataSize.getWidth() ;
  int gridHeight = dataSize.getDepth() ;
  int gridWidth = dataSize.getSize() ;
  int cropsize = 2 * radius + 1 ;
  int nthreads = cropsize * cropsize * gridHeight * gridWidth ;

  if (!backMode) {
    if (dataType == vl::vlTypeFloat) {
      cropradius_dm_kernel<float>
	<<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((float*) output.getMemory(), (const float*) data.getMemory(),
	 height, width, gridHeight, gridWidth, stride0, stride1,
         offset0, offset1, pad, radius, nthreads, rectifyBorder);
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      cropradius_dm_kernel<double>
	<<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((double*) output.getMemory(), (const double*) data.getMemory(),
	 height, width, gridHeight, gridWidth, stride0, stride1,
         offset0, offset1, pad, radius, nthreads, rectifyBorder);
#endif
    }
  } else {
    // Backward
    if (dataType == vl::vlTypeFloat) {
      cropradius_backward_dm_kernel<float>
	<<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((float*) derData.getMemory(), (const float*) derOutput.getMemory(),
	 height, width, gridHeight, gridWidth, stride0, stride1,
         offset0, offset1, pad, radius, nthreads, rectifyBorder);
    } else if (dataType == vl::vlTypeDouble) {
#ifdef ENABLE_DOUBLE
      cropradius_backward_dm_kernel<double>
	<<< vl::divideUpwards(nthreads, VL_CUDA_NUM_THREADS), VL_CUDA_NUM_THREADS >>>
	((double*) derData.getMemory(), (const double*) derOutput.getMemory(),
	 height, width, gridHeight, gridWidth, stride0, stride1,
         offset0, offset1, pad, radius, nthreads, rectifyBorder);
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
  }
}
