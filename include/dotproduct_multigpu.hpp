#ifndef __DOTPRODUCT_MULTIGPU_HPP__
#define __DOTPRODUCT_MULTIGPU_HPP__

#include<mode_vector.hpp>
#include<mesh_info.hpp>
#include<CUDA_ERRORS.hpp>
#include<thrust/reduce.h> 
#include<thrust/device_ptr.h>

#include<mpi.h>

#include<iostream>


template<typename FLOAT_TYPE, int blockSize>
__global__ void dotprod_kernel ( int order,
                                 local_quadrilateral_mesh_info<FLOAT_TYPE> m,
                                 mode_vector<FLOAT_TYPE, int> a,
                                 mode_vector<FLOAT_TYPE, int> b,
                                 FLOAT_TYPE * odata )
{

  int tid = blockDim.x*threadIdx.y + threadIdx.x;
  extern __shared__ FLOAT_TYPE sdata[];
  sdata[tid] = 0;

  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 
 
  int idx = m.compute_idx(xx, yy); 

  if ( (xx >= m.get_dimx()) ||  (yy >= m.get_dimy()) ) return;

  FLOAT_TYPE r(0);
  for(int i1 = 0; i1 < order+1; ++i1) 
    for(int i2 = 0; i2 < order+1; ++i2) 
      r += a(i1, i2, idx)*b(i1, i2, idx);

  // the next block of code came from mark harris TODO: write this credit better

  sdata[tid] = r;

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

    thrust::device_ptr<FLOAT_TYPE> d_thrust_pointer;

  public:

    dotproduct_multigpu(local_quadrilateral_mesh_info<FLOAT_TYPE> m, int _order)
      : local_mesh_info(m), order(_order)
    {

      const int dimx = local_mesh_info.get_dimx();
      const int dimy = local_mesh_info.get_dimy();
      const int blockDx = 32;
      const int blockDy = 4;

      gridSIZE = dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 );
      blockSIZE = dim3( blockDx, blockDy, 1 );

      // allocate odata
      checkError( cudaMalloc(&odata, gridSIZE.x*gridSIZE.y*sizeof(FLOAT_TYPE)) );

      thrust::device_ptr<double> tmp(odata);
      d_thrust_pointer = tmp;

    }

    ~dotproduct_multigpu()
    {
      // free odata 
      checkError( cudaFree(odata) );
    }

    FLOAT_TYPE operator() ( mode_vector<FLOAT_TYPE, int> a,
                            mode_vector<FLOAT_TYPE, int> b )
    {

      // first step kernel
      dotprod_kernel<FLOAT_TYPE, 128>
      <<< gridSIZE, blockSIZE, sizeof(FLOAT_TYPE)*129>>> // TODO: check this
      (order, local_mesh_info, a, b, odata);

      cudaThreadSynchronize();
      checkError( cudaGetLastError() );           

      FLOAT_TYPE res = thrust::reduce(d_thrust_pointer, d_thrust_pointer+(gridSIZE.x*gridSIZE.y));

//      print_odata();

#if 0
      checkError ( cudaMemcpy( &res, // to
                               odata,  // from
                               sizeof(FLOAT_TYPE),
                               cudaMemcpyDeviceToHost) );
 
      std::cout<<"___  "<<res<<std::endl;
#endif

      FLOAT_TYPE tot_res;
      MPI_Allreduce(&res, &tot_res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

      return tot_res;
    }
  

   void print_odata()
   {
     
     int pid;
     MPI_Comm_rank(MPI_COMM_WORLD, &pid);

 //    if (pid == 0)
{
     std::vector<FLOAT_TYPE> tmp(gridSIZE.x*gridSIZE.y);

     checkError ( cudaMemcpy( tmp.data(), // to
                              odata,  // from
                              tmp.size()*sizeof(FLOAT_TYPE),
                              cudaMemcpyDeviceToHost) );

     std::cerr<<"___  ";
     for (int i = 0; i < tmp.size(); ++i) 
       std::cerr<<tmp[i]<<"  ";
     std::cerr<<std::endl;
 }
   }


};




#endif
