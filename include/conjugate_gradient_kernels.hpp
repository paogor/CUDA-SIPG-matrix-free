#ifndef __CONJUGATE_GRADIENT_KERNELS_HPP__
#define __CONJUGATE_GRADIENT_KERNELS_HPP__


template<typename FLOAT_TYPE>
__global__ void alpha_kernel ( int size,
                               FLOAT_TYPE * x,
                               FLOAT_TYPE * p,
                               FLOAT_TYPE * r,
                               FLOAT_TYPE * Ap,
                               FLOAT_TYPE alpha )
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  if (idx < size)
  {
    x[idx] = x[idx] + alpha*p[idx];
    r[idx] = r[idx] - alpha*Ap[idx];
  }     

  return;
}


template<typename FLOAT_TYPE>
__global__ void beta_kernel ( int size,
                              FLOAT_TYPE * p,
                              FLOAT_TYPE * r,
                              FLOAT_TYPE beta )
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  if (idx < size) p[idx] = r[idx] + beta*p[idx];     

  return;
}

#endif

