#ifndef __CUDATIMER_HPP__
#define __CUDATIMER_HPP__

//16.10.14

//http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EVENT.html#group__CUDART__EVENT_1g14c387cc57ce2e328f6669854e6020a5


/**
 A timer for kernels.
 It measures the execution times of kernels that are launched
 between start() and stop(). 
*/
class CUDA_TIMER
{
  private:
   cudaEvent_t _start, _stop;
   float  _elapsed_millisecs;  /**< time in milliseconds with resolution
                                of around 0.5 microseconds */

  public:

   CUDA_TIMER() : _elapsed_millisecs(0) {}

   /** It starts the timer. */
   void start()
   {
     cudaEventCreate(&_start);
     cudaEventCreate(&_stop);
     cudaEventRecord(_start,0);
   }

   /** It stops the timer. */
   void stop()
   {
     cudaEventRecord(_stop,0);
     cudaEventSynchronize(_stop);

     cudaEventElapsedTime(&_elapsed_millisecs,_start,_stop);
     cudaEventDestroy(_start); 
     cudaEventDestroy(_stop);
   }

   /** It returns the milliseconds between start() and stop() */
   float elapsed_millisecs() 
   {
      return _elapsed_millisecs;
   }


};

#endif

