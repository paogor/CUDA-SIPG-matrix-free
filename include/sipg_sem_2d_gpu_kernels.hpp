#ifndef __SIPG_SEM_2D_GPU_KERNELS_HPP__
#define __SIPG_SEM_2D_GPU_KERNELS_HPP__


#include<CUDA_ERRORS.hpp>
#include<mesh_info.hpp>
#include<mode_vector.hpp>
#include<device_const.hpp>
#include<volume_gpu_kernels.hpp>



// ****************************************************************************
// ****************************** with mesh_info ******************************
// ****************************************************************************




template<typename T>
__global__ void flux_term6a ( int N,
                                   mesh_info<T>m,
                                   mode_vector<T,int> input,
                                   mode_vector<T,int> output)
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  if ( !(idx < input.get_noe()) ) return;

  T * c_lgl = device_const::lgl_weights<T>(); 

  const int idxRIGHT = m.get_neighborhood(1,idx);
  const int idxLEFT = m.get_neighborhood(3,idx);


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
__global__ void flux_term6b( int N,
                                   mesh_info<T>m,
                                   mode_vector<T,int> input,
                                   mode_vector<T,int> output)
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  if ( !(idx < input.get_noe()) ) return;

  T * c_lgl = device_const::lgl_weights<T>(); 

  const int idxDOWN = m.get_neighborhood(0,idx);
  const int idxUP = m.get_neighborhood(2,idx);


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




template<typename FLOAT_TYPE>
int mvm ( int order,
          mesh_info<FLOAT_TYPE> mesh,
          mode_vector<FLOAT_TYPE,int> input,
          mode_vector<FLOAT_TYPE,int> output ) 
{

  const int noe = input.get_noe();
  const int blockD = 128;

  volume<FLOAT_TYPE>
  <<<dim3( (noe + blockD - 1)/blockD , order+1, order+1), blockD>>>
  ( order, input, output ); 

  flux_term6a<FLOAT_TYPE>
  <<< (noe + blockD - 1)/blockD , blockD>>>
  ( order, mesh, input, output );

  flux_term6b<FLOAT_TYPE>
  <<< (noe + blockD - 1)/blockD , blockD>>>
  ( order, mesh, input, output );



#if 0

  cudaError_t error = cudaGetLastError();
  std::string lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;

#endif

  return 0;
}




// ****************************************************************************
// *********************** with quadrilateral_mesh_info ***********************
// ****************************************************************************



template<typename T>
__global__ void flux_term6a ( int N,
                                   quadrilateral_mesh_info<T>m,
                                   mode_vector<T,int> input,
                                   mode_vector<T,int> output)
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
__global__ void flux_term6b( int N,
                                   quadrilateral_mesh_info<T>m,
                                   mode_vector<T,int> input,
                                   mode_vector<T,int> output)
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



template<typename FLOAT_TYPE>
int mvm ( int order,
          quadrilateral_mesh_info<FLOAT_TYPE> mesh,
          mode_vector<FLOAT_TYPE,int> input,
          mode_vector<FLOAT_TYPE,int> output ) 
{

  const int noe = input.get_noe();
  const int blockD = 128;

  volume<FLOAT_TYPE>
  <<<dim3( (noe + blockD - 1)/blockD , order+1, order+1), blockD>>>
  ( order, input, output ); 



  const int dimx = mesh.get_dimx();
  const int dimy = mesh.get_dimy();
  const int blockDx = 32;
  const int blockDy = 4;

  flux_term6a<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
  ( order, mesh, input, output );

  flux_term6b<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
  ( order, mesh, input, output );



#if 0

  cudaError_t error = cudaGetLastError();
  std::string lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;

#endif

  return 0;
}






#endif
