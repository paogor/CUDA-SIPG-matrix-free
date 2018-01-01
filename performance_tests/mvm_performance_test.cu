#include<iostream>
#include<cstdlib>

#include<volume_gpu_kernels.hpp>
#include<sipg_sem_2d_gpu_kernels.hpp>

#include<build_square_mesh.hpp>
#include<mode_matrix_kernels.hpp>
#include<CUDA_TIMER.hpp>



template<typename T>
int mvm_test (int N, int dim, float & time)
{

  cudaError_t error;
  std::string lastError;

  load_Dphi_table<T>(N);
  load_lgl_quadrature_table<T>(N);


  square_mesh<T> sq_mesh(dim);
  host_mode_vector<T,int> h_in (dim*dim, N+1);

  for ( int i1=0; i1 < N+1; ++i1 )
    for ( int i2=0; i2 < N+1; ++i2 )
      for ( int e=0; e < dim*dim; ++e )
        h_in(i1,i2,e) = T( rand() ) / T( rand() );
 

  mode_vector<T,int> in(h_in), out(dim*dim, N+1); 

  const int blockD = 128;
  const int blockDx = 32;
  const int blockDy = 4;

  // warm up
  volume<T><<<dim3( ((dim*dim) + blockD - 1)/blockD , N+1, N+1), blockD>>>
    (N, in, out);


  CUDA_TIMER t;
  t.start();
  for(int t=0; t < 100; ++t)
  {

    volume<T><<<dim3( ((dim*dim) + blockD - 1)/blockD , N+1, N+1), blockD>>>
      (N, in, out);

    flux_term6a<T>
    <<< dim3( (dim + blockDx - 1)/blockDx, (dim + blockDy - 1)/blockDy, 1 ) ,
        dim3( blockDx, blockDy, 1 ) >>>
      (N, sq_mesh.device_info, in, out);

    flux_term6b<T>
    <<< dim3( (dim + blockDx - 1)/blockDx, (dim + blockDy - 1)/blockDy, 1 ) ,
        dim3( blockDx, blockDy, 1 ) >>>
      (N, sq_mesh.device_info, in, out);
  }

  t.stop();
  
  time = t.elapsed_millisecs()/100.0;

  cudaThreadSynchronize();

  error = cudaGetLastError();
  lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;

  in.free();
  out.free();
  sq_mesh.device_info.free();

  return 0;

}


int main(int argc, const char* argv[])
{



  for (int degree = 16; degree < 17; degree*=2)
    for (int dim = 128; dim < 2049; dim*=2)
    {
      float time = 0;
      mvm_test<double>(degree, dim, time);
      std::cout<<degree<<"\t"<<dim<<"\t"<<time<<std::endl;
    }

  cudaDeviceReset();


  return 0;

}

