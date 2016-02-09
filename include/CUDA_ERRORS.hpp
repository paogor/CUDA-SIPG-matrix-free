#ifndef __CUDA_ERRORS_HPP__
#define __CUDA_ERRORS_HPP__

#include<iostream> // for std::cerr
#include<string> 

namespace CUDA_ERRORS
{
  std::string _cudaGetErrorEnum(cudaError_t error)
  {
    switch (error)
    {
      case cudaSuccess:
        return "cudaSuccess";
      case cudaErrorMissingConfiguration:
        return "cudaErrorMissingConfiguration";
      case cudaErrorMemoryAllocation:
        return "cudaErrorMemoryAllocation";
      case cudaErrorInitializationError:
        return "cudaErrorInitializationError";
      case cudaErrorLaunchFailure:
        return "cudaErrorLaunchFailure";
      case cudaErrorPriorLaunchFailure:
        return "cudaErrorPriorLaunchFailure";
      case cudaErrorLaunchTimeout:
        return "cudaErrorLaunchTimeout";
      case cudaErrorLaunchOutOfResources:
        return "cudaErrorLaunchOutOfResources";
      case cudaErrorInvalidDeviceFunction:
        return "cudaErrorInvalidDeviceFunction";
      case cudaErrorInvalidConfiguration:
        return "cudaErrorInvalidConfiguration";
      case cudaErrorInvalidDevice:
        return "cudaErrorInvalidDevice";
      case cudaErrorInvalidValue:
        return "cudaErrorInvalidValue";
      case cudaErrorInvalidPitchValue:
        return "cudaErrorInvalidPitchValue";
      case cudaErrorInvalidSymbol:
        return "cudaErrorInvalidSymbol";
      case cudaErrorMapBufferObjectFailed:
        return "cudaErrorMapBufferObjectFailed";
      case cudaErrorUnmapBufferObjectFailed:
        return "cudaErrorUnmapBufferObjectFailed";
      case cudaErrorInvalidHostPointer:
        return "cudaErrorInvalidHostPointer";
      case cudaErrorInvalidDevicePointer:
        return "cudaErrorInvalidDevicePointer";
      case cudaErrorInvalidTexture:
        return "cudaErrorInvalidTexture";
      case cudaErrorInvalidTextureBinding:
        return "cudaErrorInvalidTextureBinding";
      case cudaErrorInvalidChannelDescriptor:
        return "cudaErrorInvalidChannelDescriptor";
      case cudaErrorInvalidMemcpyDirection:
        return "cudaErrorInvalidMemcpyDirection";
      case cudaErrorAddressOfConstant:
        return "cudaErrorAddressOfConstant";
      case cudaErrorTextureFetchFailed:
        return "cudaErrorTextureFetchFailed";
      case cudaErrorTextureNotBound:
        return "cudaErrorTextureNotBound";
      case cudaErrorSynchronizationError:
        return "cudaErrorSynchronizationError";
      case cudaErrorInvalidFilterSetting:
        return "cudaErrorInvalidFilterSetting";
      case cudaErrorInvalidNormSetting:
        return "cudaErrorInvalidNormSetting";
      case cudaErrorMixedDeviceExecution:
        return "cudaErrorMixedDeviceExecution";
      case cudaErrorCudartUnloading:
        return "cudaErrorCudartUnloading";
      case cudaErrorUnknown:
        return "cudaErrorUnknown";
      case cudaErrorNotYetImplemented:
        return "cudaErrorNotYetImplemented";
      case cudaErrorMemoryValueTooLarge:
        return "cudaErrorMemoryValueTooLarge";
      case cudaErrorInvalidResourceHandle:
        return "cudaErrorInvalidResourceHandle";
      case cudaErrorNotReady:
        return "cudaErrorNotReady";
      case cudaErrorInsufficientDriver:
        return "cudaErrorInsufficientDriver";
      case cudaErrorSetOnActiveProcess:
        return "cudaErrorSetOnActiveProcess";
      case cudaErrorInvalidSurface:
        return "cudaErrorInvalidSurface";
      case cudaErrorNoDevice:
        return "cudaErrorNoDevice";
      case cudaErrorECCUncorrectable:
        return "cudaErrorECCUncorrectable";
      case cudaErrorSharedObjectSymbolNotFound:
        return "cudaErrorSharedObjectSymbolNotFound";
      case cudaErrorSharedObjectInitFailed:
        return "cudaErrorSharedObjectInitFailed";
      case cudaErrorUnsupportedLimit:
        return "cudaErrorUnsupportedLimit";
      case cudaErrorDuplicateVariableName:
        return "cudaErrorDuplicateVariableName";
      case cudaErrorDuplicateTextureName:
        return "cudaErrorDuplicateTextureName";
      case cudaErrorDuplicateSurfaceName:
        return "cudaErrorDuplicateSurfaceName";
      case cudaErrorDevicesUnavailable:
        return "cudaErrorDevicesUnavailable";
      case cudaErrorInvalidKernelImage:
        return "cudaErrorInvalidKernelImage";
      case cudaErrorNoKernelImageForDevice:
        return "cudaErrorNoKernelImageForDevice";
      case cudaErrorIncompatibleDriverContext:
        return "cudaErrorIncompatibleDriverContext";
      case cudaErrorPeerAccessAlreadyEnabled:
        return "cudaErrorPeerAccessAlreadyEnabled";
      case cudaErrorPeerAccessNotEnabled:
        return "cudaErrorPeerAccessNotEnabled";
      case cudaErrorDeviceAlreadyInUse:
        return "cudaErrorDeviceAlreadyInUse";
      case cudaErrorProfilerDisabled:
        return "cudaErrorProfilerDisabled";
      case cudaErrorProfilerNotInitialized:
        return "cudaErrorProfilerNotInitialized";
      case cudaErrorProfilerAlreadyStarted:
        return "cudaErrorProfilerAlreadyStarted";
      case cudaErrorProfilerAlreadyStopped:
        return "cudaErrorProfilerAlreadyStopped";
#if __CUDA_API_VERSION >= 0x4000
      case cudaErrorAssert:
        return "cudaErrorAssert";
      case cudaErrorTooManyPeers:
        return "cudaErrorTooManyPeers";
      case cudaErrorHostMemoryAlreadyRegistered:
        return "cudaErrorHostMemoryAlreadyRegistered";
      case cudaErrorHostMemoryNotRegistered:
        return "cudaErrorHostMemoryNotRegistered";
#endif
      case cudaErrorStartupFailure:
        return "cudaErrorStartupFailure";
      case cudaErrorApiFailureBase:
        return "cudaErrorApiFailureBase";
    }

    return "<unknown>";
  }


  template< typename T >
  void check( T result, 
              char const *const func,
              const char *const file,
              int const line )
  {
    if (result)
    {
       std::cerr<<"CUDA ERROR "<<file<<"@"<<line<<" - ";
       std::cerr<<_cudaGetErrorEnum(result)<<"\n\t"<<func<<std::endl;

       cudaDeviceReset();
       // Make sure we call CUDA Device Reset before exiting
       exit(EXIT_FAILURE);
    }
  }

}

#define checkError(val) CUDA_ERRORS::check( (val), #val, __FILE__, __LINE__ )

#endif 
