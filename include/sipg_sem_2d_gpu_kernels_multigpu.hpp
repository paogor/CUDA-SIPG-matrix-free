#ifndef __SIPG_SEM_2D_GPU_KERNELS_MULTIGPU_HPP__
#define __SIPG_SEM_2D_GPU_KERNELS_MULTIGPU_HPP__


#include<CUDA_ERRORS.hpp>
#include<mesh_info.hpp>
#include<mode_vector.hpp>
#include<device_const.hpp>
#include<volume_gpu_kernels.hpp>



template<typename T>
__global__ void local_flux_term6a ( int N,
                                    local_quadrilateral_mesh_info<T> m,
                                    mode_vector<T,int> input,
                                    mode_vector<T,int> output )
{

  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 

  int idx = m.compute_idx(xx, yy); 

  if ( (xx >= m.get_dimx()) ||  (yy >= m.get_dimy()) ) return;

  T * c_lgl = device_const::lgl_weights<T>(); 

  const int idxRIGHT = m.get_neighborhood_RIGHT(xx, yy);
  const int idxLEFT =  m.get_neighborhood_LEFT(xx, yy);


  const T nu = 100*N*N; // check
  #pragma unroll 1     
  for (int j=0; j <= N; ++j)
  {


    T a(0), b(0); 


    const T e = c_lgl[j]*( idxLEFT<0  ?  input(0,j,idx): .5*(input(0,j,idx)  - input(N,j,idxLEFT) ));

    const T f = c_lgl[j]*( idxRIGHT<0 ?  -1*input(N,j,idx) :    .5*( input(0,j,idxRIGHT) - input(N,j,idx)) );

 
    #pragma unroll 1     
    for (int i=0; i <= N; ++i)
    {
      
      output(i,j,idx) += DPHI<T>(N,i)[0] * e + DPHI<T>(N,i)[N] * f;


      //LEFT
      a += idxLEFT<0 ?
         DPHI<T>(N,i)[0] * input(i,j,idx) :
         .5*( DPHI<T>(N,i)[0] * input(i,j,idx) + DPHI<T>(N,i)[N] *  input(i,j,idxLEFT) );

      //RIGHT
      b += idxRIGHT<0 ?
         -1*DPHI<T>(N,i)[N] * input(i,j,idx) : 
        -.5*(DPHI<T>(N,i)[N] * input(i,j,idx) + DPHI<T>(N,i)[0] * input(i,j,idxRIGHT));

    } 

      output(0,j,idx) += c_lgl[j]*a + (idxLEFT<0 ? nu: 2*nu) * e;
      output(N,j,idx) += c_lgl[j]*b - (idxRIGHT<0 ? nu: 2*nu) * f;
  

  }




  return;
}


template<typename T>
__global__ void local_flux_term6b( int N,
                                   local_quadrilateral_mesh_info<T>m,
                                   mode_vector<T,int> input,
                                   mode_vector<T,int> output )
{

  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 

  int idx = m.compute_idx(xx, yy); 

  if ( (xx >= m.get_dimx()) ||  (yy >= m.get_dimy()) ) return;

  T * c_lgl = device_const::lgl_weights<T>(); 

  const int idxDOWN = m.get_neighborhood_DOWN(xx, yy);
  const int idxUP = m.get_neighborhood_UP(xx, yy); 


  const T nu = 100*N*N; // check
  #pragma unroll 1     
  for (int j=0; j <= N; ++j)
  {

    T c(0), d(0); 

    const T g = c_lgl[j]*( idxDOWN<0 ? input(j,0,idx) : .5*(input(j,0,idx) - input(j,N,idxDOWN)));
    const T h = c_lgl[j]*( idxUP<0 ? -1*input(j,N,idx) : .5*(input(j,0,idxUP) - input(j,N,idx)) );
 
    #pragma unroll 1     
    for (int i=0; i <= N; ++i)
    {
      

      output(j,i,idx) += DPHI<T>(N,i)[0] * g + DPHI<T>(N,i)[N] * h;

      //DOWN
      c +=idxDOWN<0 ?
          DPHI<T>(N,i)[0] * input(j,i,idx) :
         .5*(DPHI<T>(N,i)[0] * input(j,i,idx) + DPHI<T>(N,i)[N] *  input(j,i,idxDOWN));

     
      //UP
      d +=idxUP<0 ?
          -1*DPHI<T>(N,i)[N] * input(j,i,idx):
         -.5*(DPHI<T>(N,i)[N] * input(j,i,idx) + DPHI<T>(N,i)[0] * input(j,i,idxUP));

    } 


      output(j,0,idx) += c_lgl[j]*c + (idxDOWN<0 ? nu: 2*nu) * g ;
      output(j,N,idx) += c_lgl[j]*d - (idxUP< 0 ? nu: 2*nu) * h ;
  

  }


  return;
}


#endif
