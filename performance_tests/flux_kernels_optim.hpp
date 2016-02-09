#ifndef __FLUX_KERNELS_OPTIM__
#define __FLUX_KERNELS_OPTIM__

#include<CUDA_ERRORS.hpp>
#include<mesh_info.hpp>
#include<mode_vector.hpp>
#include<device_const.hpp>

#include<sipg_sem_2d_gpu_kernels.hpp>



template<typename T> /* SEM SIPG */
__global__ void penalty_terms_in_diagonal_blocks ( int N,
                                                   quadrilateral_mesh_info<T> m,
                                                   mode_vector<T, int> input, 
                                                   mode_vector<T, int> output )
{

  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 

  int idx = m.compute_idx(xx, yy); 

  if ( (xx >= m.get_dimx()) ||  (yy >= m.get_dimy()) ) return;

  T * c_lgl = device_const::lgl_weights<T>(); 

  const T nu = 100*N*N; // check
  const T hF = 1./sqrtf(input.get_noe()); // this will change 

  for (int i = 0; i <= N ; ++i)
  { 
    const T r1 = nu/hF;
    output(i,0,idx) +=  r1 * input(i,0,idx) * c_lgl[i] /* .5*/ * hF ; 
 
    const T r2 = nu/hF; 
    output(i,N,idx) +=  r2 * input(i,N,idx) * c_lgl[i] /* .5*/ * hF ; 
   
    const T r3 = nu/hF;
    output(0,i,idx) +=  r3 * input(0,i,idx) * c_lgl[i] /* .5*/ * hF ; 

    const T r4 = nu/hF;    
    output(N,i,idx) +=  r4 * input(N,i,idx) * c_lgl[i] /* .5*/ * hF ; 
  }

  return;

}




template<typename T> /* SEM SIPG */
__global__ void penalty_terms_in_extra_diagonal_blocks (int N,
                                                        quadrilateral_mesh_info<T> m,
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
  const int idxRIGHT = m.get_neighborhood_RIGHT(xx, yy);
  const int idxLEFT =  m.get_neighborhood_LEFT(xx, yy);

  const T nu = 100*N*N; // check
  const T hF = 1./sqrtf(input.get_noe()); // this will change 


  for (int i = 0; i <= N ; ++i)
  { 
    const T r1 = nu/hF;    
    if ( m.get_neighborhood_LEFT(xx, yy) >= 0 )
     output(0,i,idx) -=  r1 * input(N,i,idxLEFT) * c_lgl[i] /* .5*/ * hF ;
 
    const T r2 = nu/hF; 
    if ( m.get_neighborhood_RIGHT(xx, yy) >= 0 )
     output(N,i,idx) -=  r2 * input(0,i,idxRIGHT) * c_lgl[i] /* .5*/ * hF ; 

    const T r3 = nu/hF;
    if ( m.get_neighborhood_UP(xx, yy) >= 0 )
     output(i,N,idx) -=  r3 * input(i,0,idxUP) * c_lgl[i] /* .5*/ * hF ;

    const T r4 = nu/hF;    
    if ( m.get_neighborhood_DOWN(xx, yy) >= 0 )
     output(i,0,idx) -=  r4 * input(i,N,idxDOWN) * c_lgl[i] /* .5*/ * hF ;
  }

  return;

}

template<typename T>
__global__ void diagonal_flux_term ( int N,
                                   quadrilateral_mesh_info<T>m,
                                   mode_vector<T,int> input,
                                   mode_vector<T,int> output)
{

  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 

  int idx = m.compute_idx(xx, yy); 

  if ( (xx >= m.get_dimx()) ||  (yy >= m.get_dimy()) ) return;

  T * c_lgl = device_const::lgl_weights<T>(); 

  const T nDOWN  = m.get_neighborhood_DOWN(xx, yy)<0 ? 1 : .5 ;
  const T nUP    = m.get_neighborhood_UP(xx, yy)<0 ? 1 : .5 ;
  const T nRIGHT = m.get_neighborhood_RIGHT(xx, yy)<0 ? 1 : .5 ;
  const T nLEFT  = m.get_neighborhood_LEFT(xx, yy)<0 ? 1 : .5 ;


  #pragma unroll 1     
  for (int j=0; j <= N; ++j)
  #pragma unroll 1     
    for (int i=0; i <= N; ++i)
    {
      //LEFT
      output(0,j,idx) -= -1*c_lgl[j]*DPHI<T>(N,i)[0] * input(i,j,idx) * nLEFT ;
      output(i,j,idx) -= -1*c_lgl[j]*DPHI<T>(N,i)[0] * input(0,j,idx) * nLEFT ;

      //RIGHT
      output(N,j,idx) -= c_lgl[j]*DPHI<T>(N,i)[N] * input(i,j,idx) * nRIGHT ;
      output(i,j,idx) -= c_lgl[j]*DPHI<T>(N,i)[N] * input(N,j,idx) * nRIGHT ;

      //DOWN
      output(j,0,idx) -= -1*c_lgl[j]*DPHI<T>(N,i)[0] * input(j,i,idx) * nDOWN;
      output(j,i,idx) -= -1*c_lgl[j]*DPHI<T>(N,i)[0] * input(j,0,idx) * nDOWN;

      //UP
      output(j,N,idx) -= c_lgl[j]*DPHI<T>(N,i)[N] * input(j,i,idx) * nUP;
      output(j,i,idx) -= c_lgl[j]*DPHI<T>(N,i)[N] * input(j,N,idx) * nUP;
    } 


  return;
}





template<typename FLOAT_TYPE>
__global__ void extra_diagonal_flux_term ( int N,
                                   quadrilateral_mesh_info<FLOAT_TYPE> m,
                                   mode_vector<FLOAT_TYPE,int> input,
                                   mode_vector<FLOAT_TYPE,int> output)
{


  int xx = blockDim.x*blockIdx.x + threadIdx.x; 
  int yy = blockDim.y*blockIdx.y + threadIdx.y; 

  int idx = m.compute_idx(xx, yy); 

  if ( (xx >= m.get_dimx()) ||  (yy >= m.get_dimy()) ) return;

  FLOAT_TYPE * c_lgl = device_const::lgl_weights<FLOAT_TYPE>(); 

  const int idxDOWN = m.get_neighborhood_DOWN(xx, yy);
  const int idxUP = m.get_neighborhood_UP(xx, yy); 
  const int idxRIGHT = m.get_neighborhood_RIGHT(xx, yy);
  const int idxLEFT =  m.get_neighborhood_LEFT(xx, yy);


  #pragma unroll 1     
  for (int j=0; j <= N; ++j)
  #pragma unroll 1     
    for (int i=0; i <= N; ++i)
    {

      if (!(idxLEFT < 0)) // this if prevents the access to unallocated memory
      {
        //LEFT
        output(0,j,idx) +=  .5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[N] * input(i,j,idxLEFT);
        output(i,j,idx) += -.5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[0] * input(N,j,idxLEFT);
      }


      if (!(idxRIGHT < 0))
      {
        //RIGHT
        output(N,j,idx) += -.5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[0] * input(i,j,idxRIGHT);
        output(i,j,idx) +=  .5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[N] * input(0,j,idxRIGHT);
      }


      if (!(idxDOWN< 0))
      {
        //DOWN
        output(j,0,idx) +=  .5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[N] * input(j,i,idxDOWN);
        output(j,i,idx) += -.5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[0] * input(j,N,idxDOWN);
      }


      if (!(idxUP< 0))
      {
        //UP
        output(j,N,idx) += -.5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[0] * input(j,i,idxUP);
        output(j,i,idx) +=  .5 * c_lgl[j] * DPHI<FLOAT_TYPE>(N,i)[N] * input(j,0,idxUP);
      }

    } 


  return;
}






template<typename FLOAT_TYPE>
int mvm0 ( int order,
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


  diagonal_flux_term<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
   ( order, mesh, input, output );

  extra_diagonal_flux_term<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
   ( order, mesh, input, output );

  penalty_terms_in_diagonal_blocks<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
   ( order, mesh, input, output );

  penalty_terms_in_extra_diagonal_blocks<FLOAT_TYPE>
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


template<typename FLOAT_TYPE>
int test_flux0 ( int order,
                 quadrilateral_mesh_info<FLOAT_TYPE> mesh,
                 mode_vector<FLOAT_TYPE,int> input,
                 mode_vector<FLOAT_TYPE,int> output ) 
{

  const int dimx = mesh.get_dimx();
  const int dimy = mesh.get_dimy();
  const int blockDx = 32;
  const int blockDy = 4;


  diagonal_flux_term<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
   ( order, mesh, input, output );

  extra_diagonal_flux_term<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
   ( order, mesh, input, output );

  penalty_terms_in_diagonal_blocks<FLOAT_TYPE>
  <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
      dim3( blockDx, blockDy, 1 ) >>>
   ( order, mesh, input, output );

  penalty_terms_in_extra_diagonal_blocks<FLOAT_TYPE>
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



// ********************************************************************************************



template<typename T>
__global__ void flux_term8 ( int N,
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
  const int idxRIGHT = m.get_neighborhood_RIGHT(xx, yy);
  const int idxLEFT =  m.get_neighborhood_LEFT(xx, yy);


  const T nu = 100*N*N; // check
  #pragma unroll 1     
  for (int j=0; j <= N; ++j)
  {


    T a(0), b(0); 


    const T e = c_lgl[j]*( idxLEFT<0  ?  input(0,j,idx): .5*(input(0,j,idx)  - input(N,j,idxLEFT) ));
    const T f = c_lgl[j]*( idxRIGHT<0 ?  -1*input(N,j,idx) :    .5*( input(0,j,idxRIGHT) - input(N,j,idx)) );

    T c(0), d(0); 

    const T g = c_lgl[j]*( idxDOWN<0 ? input(j,0,idx) : .5*(input(j,0,idx) - input(j,N,idxDOWN)));
    const T h = c_lgl[j]*( idxUP<0 ? -1*input(j,N,idx) : .5*(input(j,0,idxUP) - input(j,N,idx)) );
 
 
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

      output(0,j,idx) += c_lgl[j]*a + (idxLEFT<0 ? nu: 2*nu) * e;
      output(N,j,idx) += c_lgl[j]*b - (idxRIGHT<0 ? nu: 2*nu) * f;
      output(j,0,idx) += c_lgl[j]*c + (idxDOWN<0 ? nu: 2*nu) * g ;
      output(j,N,idx) += c_lgl[j]*d - (idxUP< 0 ? nu: 2*nu) * h ;
   

  }




  return;
}





template<typename FLOAT_TYPE>
int mvm8 ( int order,
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

  flux_term8<FLOAT_TYPE>
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



template<typename FLOAT_TYPE>
int test_flux8 ( int order,
                 quadrilateral_mesh_info<FLOAT_TYPE> mesh,
                 mode_vector<FLOAT_TYPE,int> input,
                 mode_vector<FLOAT_TYPE,int> output ) 
{

  const int dimx = mesh.get_dimx();
  const int dimy = mesh.get_dimy();
  const int blockDx = 32;
  const int blockDy = 4;

  flux_term8<FLOAT_TYPE>
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


template<typename FLOAT_TYPE>
int test_flux ( int order,
                quadrilateral_mesh_info<FLOAT_TYPE> mesh,
                mode_vector<FLOAT_TYPE,int> input,
                mode_vector<FLOAT_TYPE,int> output ) 
{

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
