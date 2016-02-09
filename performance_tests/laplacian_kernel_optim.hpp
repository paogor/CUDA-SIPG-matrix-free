#ifndef __SEM_GPU_LAPLACIAN_OPTIM_HPP__
#define __SEM_GPU_LAPLACIAN_OPTIM_HPP__


#include<device_const.hpp>
using device_const::lgl_weights;


inline __host__ __device__ int modeINDEX (int N, int totalSize, int i1, int i2)
{
  return (i1*(N+1)+i2)*totalSize;
}


/*
template<typename T>
__global__ void volume (int N, T * input, T * output) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 
  int TOTALsize = blockDim.x*gridDim.x;

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  // i1,i2 output modes

  T result = 0.;

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result += lgl_weights<T>()[k]*( lgl_weights<T>()[i2]*Dphi<T>(N, i1, k)*Dphi<T>(N, i1, k)
                              + lgl_weights<T>()[i1]*Dphi<T>(N, i2, k)*Dphi<T>(N, i2, k) );

  result *= input[modeINDEX(N,TOTALsize,i1,i2) + idx];

  // add HERE Mass



  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  for (int i=0; i <= N; ++i)
    if ( i != i1 )
    {
      T r = 0;

      #pragma unroll 1     
      for (int k=0; k <= N; ++k)
        r += lgl_weights<T>()[k] * Dphi<T>(N, i, k)*Dphi<T>(N, i1, k) ; result +=  lgl_weights<T>()[i2]*r*input[modeINDEX(N, TOTALsize,i,i2) + idx]; 
    }
   


  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  for (int i=0; i <= N; ++i)
    if ( i != i2 )
    {
      T r = 0;

      #pragma unroll 1 
      for (int k=0; k <= N; ++k)
        r += lgl_weights<T>()[k]* Dphi<T>(N, i2, k)*Dphi<T>(N, i, k) ;

      result += lgl_weights<T>()[i1]*r*input[modeINDEX(N,TOTALsize,i1,i) + idx];

    }


  output[modeINDEX(N,TOTALsize,i1,i2) + idx] = result;

  return;

}

*/





template<typename T>
__global__ void volume0 ( int N,
                         mode_vector<T, int> input,
                         mode_vector<T, int> output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  //  i1,i2 output modes
 
  T result = 0.;

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
   result += lgl_weights<T>()[k]*( lgl_weights<T>()[i2]*Dphi<T>(N, i1, k)*Dphi<T>(N, i1, k)
             + lgl_weights<T>()[i1]*Dphi<T>(N, i2, k)*Dphi<T>(N, i2, k) );

  result *= input(i1,i2,idx);

  // add HERE Mass

  // ([0..N]/i1,i2;i1,i2) extra-diagonal term of Laplacian operator
  // It takes [0..N],i2 as INPUT.


  for (int i=0; i <= N; ++i)
    if ( i != i1 )
    {
       T r = 0;

       #pragma unroll 1     
       for (int k=0; k <= N; ++k)
         r += lgl_weights<T>()[k] * Dphi<T>(N, i, k)*Dphi<T>(N, i1, k) ;

       result +=  lgl_weights<T>()[i2]*r*input(i,i2,idx);

    }
                                                                                                  


  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  for (int i=0; i <= N; ++i)
    if ( i != i2 )
    {
      T r = 0;

      #pragma unroll 1 

      for (int k=0; k <= N; ++k)
        r += lgl_weights<T>()[k]* Dphi<T>(N, i2, k)*Dphi<T>(N, i, k) ;

      result += lgl_weights<T>()[i1]*r*input(i1,i,idx);

    }


   output(i1,i2,idx) = result;
   return;

}







template<typename T>
__global__ void volume1 ( int N,
                         mode_vector<T, int> input,
                         mode_vector<T, int> output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;



  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 
  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);



  // i1,i2 output modes

  T result = 0;

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 


  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result += c_lgl[k]*( c_lgl[i2]*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl[i1]*c_Dphi2[k]*c_Dphi2[k] );

  result *= input(i1,i2,idx);

  // add HERE Mass



  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  for (int i=0; i < i1; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
      r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result +=  c_lgl[i2]*r*input(i,i2,idx);

  }
   

  for (int i = i1 + 1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
      r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result +=  c_lgl[i2]*r*input(i,i2,idx);

  }

  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  for (int i=0; i < i2; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result += c_lgl[i1]*r*input(i1,i,idx);

  }


  for (int i=i2+1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result += c_lgl[i1]*r*input(i1,i,idx);

  }
   

  output(i1,i2,idx) = result;

  return;

}




template<typename T>
__global__ void volume2 ( int N,
                         mode_vector<T, int> input,
                         mode_vector<T, int> output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;


  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 

  T c_lgl_i1 = c_lgl[i1];
  T c_lgl_i2 = c_lgl[i2];

  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);


  // i1,i2 output modes

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  T result0 = 0;

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result0 += c_lgl[k]*( c_lgl_i2*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl_i1*c_Dphi2[k]*c_Dphi2[k] );

  result0 *= input(i1,i2,idx);

  // add HERE Mass



  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  T result1 = 0;

  for (int i=0; i < i1; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1     
     for (int k=0; k <= N; ++k)
       r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result1 +=  r*input(i,i2,idx);

  }
   

  for (int i = i1 + 1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
      r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result1 +=  r*input(i,i2,idx);

  }

  result1 *= c_lgl_i2;

  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  T result2 = 0;


  for (int i=0; i < i2; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result2 += r*input(i1,i,idx);

  }


  for (int i=i2+1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result2 += r*input(i1,i,idx);

  }
   

  result2 *= c_lgl_i1;

  output(i1,i2,idx) = result0 + result1 + result2;

  return;

}



template<typename T>
__global__ void volume3 ( int N,
                         T * input,
                         T * output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  int TOTALsize = blockDim.x*gridDim.x;


  // i1,i2 output modes


  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 

  T c_lgl_i1 = c_lgl[i1];
  T c_lgl_i2 = c_lgl[i2];

  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);


  // compute on thread 0 the index of mode i1,i2
  __shared__ int mode_index_i1_i2;
  if (threadIdx.x == 0 )  mode_index_i1_i2 =  modeINDEX(N,TOTALsize,i1,i2); 
  __syncthreads();


  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  T result0 = 0;

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result0 += c_lgl[k]*( c_lgl_i2*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl_i1*c_Dphi2[k]*c_Dphi2[k] );

  result0 *= input[mode_index_i1_i2 + idx]; 

  // add HERE Mass



  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  T result1 = 0;

  __shared__ int mode_index_i_i2;

  for (int i=0; i < i1; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i_i2 =  modeINDEX(N,TOTALsize,i,i2); 
   __syncthreads();

    #pragma unroll 1     
     for (int k=0; k <= N; ++k)
       r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result1 +=  r*input[mode_index_i_i2 + idx];
  }


  for (int i = i1 + 1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i_i2 =  modeINDEX(N,TOTALsize,i,i2); 
   __syncthreads();

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
      r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result1 +=  r*input[mode_index_i_i2 + idx];
  }



  result1 *= c_lgl_i2;

  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  T result2 = 0;

  __shared__ int mode_index_i1_i;

  for (int i=0; i < i2; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i1_i =  modeINDEX(N,TOTALsize,i1,i); 
   __syncthreads();


    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result2 += r*input [ mode_index_i1_i + idx];
  }


  for (int i=i2+1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i1_i =  modeINDEX(N,TOTALsize,i1,i); 
   __syncthreads();

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result2 += r*input [ mode_index_i1_i + idx];
  }
   

  result2 *= c_lgl_i1;

  output[mode_index_i1_i2 + idx] = result0 + result1 + result2;

  return;

}





template<typename T>
__global__ void volume4 ( int N,
                         T * input,
                         T * output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  int TOTALsize = blockDim.x*gridDim.x;


  // i1,i2 output modes


  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 

  T c_lgl_i1 = c_lgl[i1];
  T c_lgl_i2 = c_lgl[i2];

  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);

  const int warpSize = 32;

  // compute on thread 0 the index of mode i1,i2
  __shared__ int mode_index_i1_i2[warpSize];
  if (threadIdx.x < warpSize )  mode_index_i1_i2[threadIdx.x] =  modeINDEX(N,TOTALsize,i1,i2); 
  __syncthreads();

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  T result0 = 0;

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result0 += c_lgl[k]*( c_lgl_i2*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl_i1*c_Dphi2[k]*c_Dphi2[k] );

  result0 *= input[mode_index_i1_i2[threadIdx.x%warpSize] + idx]; 

  // add HERE Mass



  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  T result1 = 0;

  __shared__ int mode_index_i_i2[warpSize];

  for (int i=0; i < i1; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x < warpSize ) mode_index_i_i2[threadIdx.x] =  modeINDEX(N,TOTALsize,i,i2); 
   __syncthreads();

    #pragma unroll 1     
     for (int k=0; k <= N; ++k)
       r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result1 +=  r*input[mode_index_i_i2[threadIdx.x%warpSize] + idx];
  }


  for (int i = i1 + 1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x < warpSize ) mode_index_i_i2[threadIdx.x] =  modeINDEX(N,TOTALsize,i,i2); 
   __syncthreads();

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
      r += c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ;

    result1 +=  r*input[mode_index_i_i2[threadIdx.x%warpSize] + idx];
  }



  result1 *= c_lgl_i2;

  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  T result2 = 0;

  __shared__ int mode_index_i1_i[warpSize];

  for (int i=0; i < i2; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x < warpSize ) mode_index_i1_i[threadIdx.x] =  modeINDEX(N,TOTALsize,i1,i); 
   __syncthreads();


    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result2 += r*input [ mode_index_i1_i[threadIdx.x%warpSize] + idx];
  }


  for (int i=i2+1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x < warpSize ) mode_index_i1_i[threadIdx.x] =  modeINDEX(N,TOTALsize,i1,i); 
   __syncthreads();

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
      r += c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;

    result2 += r*input [ mode_index_i1_i[threadIdx.x%warpSize] + idx];
  }
   

  result2 *= c_lgl_i1;

  output[mode_index_i1_i2[threadIdx.x%warpSize] + idx] = result0 + result1 + result2;

  return;

}




template<typename T>
__global__ void volume5 ( int N,
                         T * input,
                         T * output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  int TOTALsize = blockDim.x*gridDim.x;


  // i1,i2 output modes


  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 

  T c_lgl_i1 = c_lgl[i1];
  T c_lgl_i2 = c_lgl[i2];

  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);


  // compute on thread 0 the index of mode i1,i2
  __shared__ int mode_index_i1_i2;
  if (threadIdx.x == 0 )  mode_index_i1_i2 =  modeINDEX(N,TOTALsize,i1,i2); 
  __syncthreads();


  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  T result0 = 0;

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result0 += c_lgl[k]*( c_lgl_i2*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl_i1*c_Dphi2[k]*c_Dphi2[k] );

  result0 *= input[mode_index_i1_i2 + idx]; 

  // add HERE Mass



  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  T result1 = 0;

  __shared__ int mode_index_i_i2;
  __shared__ T basis;

  for (int i=0; i < i1; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i_i2 =  modeINDEX(N,TOTALsize,i,i2); 
   __syncthreads();

    #pragma unroll 1     
     for (int k=0; k <= N; ++k)
     {

       __syncthreads();
       if ( threadIdx.x == 0 ) basis =  c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ; 
       __syncthreads();

       r += basis;
     }

    result1 +=  r*input[mode_index_i_i2 + idx];
  }


  for (int i = i1 + 1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i_i2 =  modeINDEX(N,TOTALsize,i,i2); 
   __syncthreads();

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
    {

       __syncthreads();
       if ( threadIdx.x == 0 ) basis =  c_lgl[k] * c_DphiI[k]*c_Dphi1[k] ; 
       __syncthreads();

      r += basis;

    }

    result1 +=  r*input[mode_index_i_i2 + idx];
  }



  result1 *= c_lgl_i2;

  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  T result2 = 0;

  __shared__ int mode_index_i1_i;

  for (int i=0; i < i2; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i1_i =  modeINDEX(N,TOTALsize,i1,i); 
   __syncthreads();


    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
     {

       __syncthreads();
       if ( threadIdx.x == 0 ) basis = c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;
       __syncthreads();

      r += basis;

    }

    result2 += r*input [ mode_index_i1_i + idx];
  }


  for (int i=i2+1; i <= N; ++i)
  {
    T r = 0;

    T * c_DphiI = DPHI<T>(N,i);

   __syncthreads();
   if ( threadIdx.x == 0 ) mode_index_i1_i =  modeINDEX(N,TOTALsize,i1,i); 
   __syncthreads();

    #pragma unroll 1 
    for (int k=0; k <= N; ++k)
     {

       __syncthreads();
       if ( threadIdx.x == 0 ) basis = c_lgl[k]* c_Dphi2[k]*c_DphiI[k] ;
       __syncthreads();

      r += basis;

    }


    result2 += r*input [ mode_index_i1_i + idx];
  }
   

  result2 *= c_lgl_i1;

  output[mode_index_i1_i2 + idx] = result0 + result1 + result2;

  return;

}



template<typename T>
__global__ void volume5 ( int N,
                         mode_vector<T, int> input,
                         mode_vector<T, int> output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  //  i1,i2 output modes
 
  T result = 0;

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
   result += lgl_weights<T>()[k]*( lgl_weights<T>()[i2]*Dphi<T>(N, i1, k)*Dphi<T>(N, i1, k)
             + lgl_weights<T>()[i1]*Dphi<T>(N, i2, k)*Dphi<T>(N, i2, k) );

  result *= input(i1,i2,idx);

  // add HERE Mass


  for (int i=0; i <= N; ++i)
  {

    // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
    // It takes [0..N],i2 as INPUT.

    if ( i != i1 )
    {
       T r = 0;

       #pragma unroll 1     
       for (int k=0; k <= N; ++k)
         r += lgl_weights<T>()[k] * Dphi<T>(N, i, k)*Dphi<T>(N, i1, k) ;

       result +=  lgl_weights<T>()[i2]*r*input(i,i2,idx);

    }
                                                                                                  


    // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
    // It takes i1,[0..N] as INPUT.


    if ( i != i2 )
    {
      T r = 0;

      #pragma unroll 1 

      for (int k=0; k <= N; ++k)
        r += lgl_weights<T>()[k]* Dphi<T>(N, i2, k)*Dphi<T>(N, i, k) ;

      result += lgl_weights<T>()[i1]*r*input(i1,i,idx);

    }

  }  

   output(i1,i2,idx) = result;
   return;

}


template<typename T>
__global__ void volume6 ( int N,
                         mode_vector<T, int> input,
                         mode_vector<T, int> output ) // for now only Laplacian
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  if ( !(idx < input.get_noe()) ) return;

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 
  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);

  //  i1,i2 output modes
 
  T result = 0;

  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result += c_lgl[k]*( c_lgl[i2]*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl[i1]*c_Dphi2[k]*c_Dphi2[k] );


  result *= input(i1,i2,idx);

  // add HERE Mass


    #pragma unroll 1     
  for (int i=0; i <= N; ++i)
  {

    T r1 = 0;
    T r2 = 0;

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
    {

     const T a = c_lgl[k] * c_DphiI[k];

    // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
    // It takes [0..N],i2 as INPUT.


      r1 += /*c_lgl[k] * c_DphiI[k]*/ a * c_Dphi1[k];


    // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
    // It takes i1,[0..N] as INPUT.

      r2 += /*c_lgl[k] * c_DphiI[k]*/ a * c_Dphi2[k];


    }


    result +=   c_lgl[i2]*r1*input(i,i2,idx)*( i != i1 ? T(1): T(0) )
              + c_lgl[i1]*r2*input(i1,i,idx)*( i != i2 ? T(1): T(0) );

  }


   output(i1,i2,idx) = result;
   return;

}




/* This is slower than the previous implementation  */

template<typename T>
__global__ void volume8 ( int N,
                         T * input,
                         T * output ) // for now only Laplacian
{

  const int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  const int i1 = blockIdx.y;
  const int i2 = blockIdx.z;

  T * c_lgl = lgl_weights<T>(); // pointer to LGL weights in device constant memory 
  T * c_Dphi1 = DPHI<T>(N, i1);
  T * c_Dphi2 = DPHI<T>(N, i2);

  //  i1,i2 output modes

  const int noe = gridDim.x * blockDim.x;

  int mode_index_i2 =  modeINDEX(N,noe,0,i2); 
  int mode_index_i1 =  modeINDEX(N,noe,i1,0); 

  const int _next_i1 = noe*(N+1);

  T result = 0;

  #pragma unroll 1     
  for (int i=0; i <= N; ++i)
  {

    T r1 = T(0);
    T r2 = T(0);

    T * c_DphiI = DPHI<T>(N,i);

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
    {

     const T a = c_lgl[k] * c_DphiI[k];

    // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
    // It takes [0..N],i2 as INPUT.


      r1 += a * c_Dphi1[k];


    // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
    // It takes i1,[0..N] as INPUT.

      r2 += a * c_Dphi2[k];


    }


    result +=   c_lgl[i2]*r1*input[mode_index_i2 + idx]//*( i != i1 ? T(1): T(0) )
              + c_lgl[i1]*r2*input[mode_index_i1 + idx];//*( i != i2 ? T(1): T(0) );

    mode_index_i2 += _next_i1; // next i1
    mode_index_i1 += noe;      // next i2

  }

   output[modeINDEX(N,noe,i1,i2) + idx] = result;
   return;

}


#endif
