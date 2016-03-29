#ifndef __DOTPRODUCT_MULTIGPU_HPP__
#define __DOTPRODUCT_MULTIGPU_HPP__



//#define USE_THRUST

#ifdef USE_THRUST
  #include<thrust/reduce.h> 
  #include<thrust/device_ptr.h>
#endif

#include<mode_vector.hpp>
#include<mesh_info.hpp>
#include<CUDA_ERRORS.hpp>

#ifndef __DEBUG_NO_MPI__
  #include<mpi.h>
#endif

#include<iostream>


template<typename FLOAT_TYPE, int blockSize>
__global__ void dotprod_kernel ( int order,
                                 local_quadrilateral_mesh_info<FLOAT_TYPE> m,
                                 mode_vector<FLOAT_TYPE, int> a,
                                 mode_vector<FLOAT_TYPE, int> b,
                                 FLOAT_TYPE * odata )
{

  int tid = blockDim.x*threadIdx.y + threadIdx.x;
  __shared__ FLOAT_TYPE sdata[blockSize];
  sdata[tid] = 0;

  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 
 

  if ( (xx < m.get_dimx()) && (yy < m.get_dimy()) )
  { 

    int idx = m.compute_idx(xx, yy); 

    FLOAT_TYPE r(0);
    for(int i1 = 0; i1 < order+1; ++i1) 
      for(int i2 = 0; i2 < order+1; ++i2) 
        r += a(i1, i2, idx)*b(i1, i2, idx);

    sdata[tid] = r;

  }

  // the next block of code came from mark harris TODO: write this credit better
  __syncthreads();

  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] += sdata[tid + 512]; } __syncthreads(); }

  if (blockSize >= 512)  { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }

  if (blockSize >= 256)  { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }

  if (blockSize >= 128)  { if (tid <  64) { sdata[tid] += sdata[tid +  64]; } __syncthreads(); }

  if (tid < 32)
  {
    volatile FLOAT_TYPE * v_sdata = sdata;    
    if (blockSize >=  64) v_sdata[tid] += v_sdata[tid + 32];
    if (blockSize >=  32) v_sdata[tid] += v_sdata[tid + 16];
    if (blockSize >=  16) v_sdata[tid] += v_sdata[tid +  8];
    if (blockSize >=   8) v_sdata[tid] += v_sdata[tid +  4];
    if (blockSize >=   4) v_sdata[tid] += v_sdata[tid +  2];
    if (blockSize >=   2) v_sdata[tid] += v_sdata[tid +  1];
  }

  // END in-block reduction

  if (tid == 0) odata[gridDim.x*blockIdx.y + blockIdx.x] = sdata[0];

  return;

}






template <typename FLOAT_TYPE>
class dotproduct_multigpu
{

  private:

    int order;
    FLOAT_TYPE * odata;
    local_quadrilateral_mesh_info<FLOAT_TYPE> local_mesh_info;

    dim3 gridSIZE;
    dim3 blockSIZE;

#ifdef USE_THRUST
    thrust::device_ptr<FLOAT_TYPE> d_thrust_begin;
    thrust::device_ptr<FLOAT_TYPE> d_thrust_end;
#else
    std::vector<FLOAT_TYPE> host_odata;
#endif

  public:

    dotproduct_multigpu(local_quadrilateral_mesh_info<FLOAT_TYPE> m, int _order)
      : local_mesh_info(m), order(_order)
    {

      const int dimx = local_mesh_info.get_dimx();
      const int dimy = local_mesh_info.get_dimy();
      const int blockDx = 32;
      const int blockDy = 32;

      gridSIZE = dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 );
      blockSIZE = dim3( blockDx, blockDy, 1 );

      // allocate odata
      checkError( cudaMalloc(&odata, gridSIZE.x*gridSIZE.y*sizeof(FLOAT_TYPE)) );


#ifdef USE_THRUST
      thrust::device_ptr<FLOAT_TYPE> tmp(odata);
      d_thrust_begin = tmp;
      d_thrust_end = tmp + (gridSIZE.x*gridSIZE.y);
#else
      host_odata.resize(gridSIZE.x*gridSIZE.y);
#endif

    }

    ~dotproduct_multigpu()
    {
      // free odata 
      checkError( cudaFree(odata) );
    }

    double operator() ( mode_vector<FLOAT_TYPE, int> a,
                        mode_vector<FLOAT_TYPE, int> b )
    {

      // first step kernel
      dotprod_kernel<FLOAT_TYPE, 1024>
      <<< gridSIZE, blockSIZE>>> // TODO: check this
      (order, local_mesh_info, a, b, odata);

      double tot_res(0), res(0);

#ifdef USE_THRUST

      res = thrust::reduce(d_thrust_begin, d_thrust_end);

#else

      checkError ( cudaMemcpy( host_odata.data(), // to
                               odata,  // from
                               host_odata.size()*sizeof(FLOAT_TYPE),
                               cudaMemcpyDeviceToHost) );

      for(int i = 0; i < host_odata.size(); ++i)
        res += host_odata[i];

#endif

 
#ifndef __DEBUG_NO_MPI__

      MPI_Allreduce(&res, &tot_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

#else

      tot_res = res;

#endif

      return tot_res;

    }
  
};




#endif
