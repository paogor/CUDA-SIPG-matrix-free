#ifndef __VOLUME_GPU_KERNELS_HPP__
#define __VOLUME_GPU_KERNELS_HPP__

#include<mode_vector.hpp>
#include<device_const.hpp>



template<typename FLOAT_TYPE> // for now only Laplacian
__global__ void volume ( int N,
                         mode_vector<FLOAT_TYPE, int> input,
                         mode_vector<FLOAT_TYPE, int> output )
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  if ( !(idx < input.get_noe()) ) return;

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  /* pointer to LGL weights in device constant memory */
  FLOAT_TYPE * c_lgl = device_const::lgl_weights<FLOAT_TYPE>();

  FLOAT_TYPE * c_Dphi1 = DPHI<FLOAT_TYPE>(N, i1);
  FLOAT_TYPE * c_Dphi2 = DPHI<FLOAT_TYPE>(N, i2);

  //  i1,i2 output modes

  FLOAT_TYPE result = 0;
/*
  // (i1,i2;i1,i2) diagonal term: Laplacian, Mass 
  // It takes i1,i2 as INPUT. 

  #pragma unroll 1  // prevent loop unroll to lower register usage   
  for (int k=0; k <= N; ++k)
    result += c_lgl[k]*( c_lgl[i2]*c_Dphi1[k]*c_Dphi1[k]
                       + c_lgl[i1]*c_Dphi2[k]*c_Dphi2[k] );


  result *= input(i1,i2,idx);

  // add HERE Mass
*/

  #pragma unroll 1     
  for (int i=0; i <= N; ++i)
  {

    FLOAT_TYPE r1 = FLOAT_TYPE(0);
    FLOAT_TYPE r2 = FLOAT_TYPE(0);

    FLOAT_TYPE * c_DphiI = DPHI<FLOAT_TYPE>(N,i);

    #pragma unroll 1     
    for (int k=0; k <= N; ++k)
    {

     const FLOAT_TYPE a = c_lgl[k] * c_DphiI[k];

    // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
    // It takes [0..N],i2 as INPUT.


      r1 += a * c_Dphi1[k];


    // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
    // It takes i1,[0..N] as INPUT.

      r2 += a * c_Dphi2[k];


    }


    result +=   c_lgl[i2]*r1*input(i,i2,idx)//*( i != i1 ? FLOAT_TYPE(1): FLOAT_TYPE(0) )
              + c_lgl[i1]*r2*input(i1,i,idx);//*( i != i2 ? FLOAT_TYPE(1): FLOAT_TYPE(0) );

  }


   output(i1,i2,idx) = result;
   return;

}



#endif 
