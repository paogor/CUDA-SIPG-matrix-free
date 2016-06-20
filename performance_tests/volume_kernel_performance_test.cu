#include<iostream>
#include<cstdlib>

#include<volume_gpu_kernels.hpp>
#include"laplacian_kernel_optim.hpp"

#include<mode_matrix_kernels.hpp>

#if 1
  #define TEST_P
#else
  #define TEST_H
#endif


template<typename T>
int performance_test (int N, int blockD, int num_of_el)
{

  cudaError_t error;
  std::string lastError;

  load_Dphi_table<T>(N);
  load_lgl_quadrature_table<T>(N);


  host_mode_vector<T,int> h_in (num_of_el, N+1);

  for ( int i1=0; i1 < N+1; ++i1 )
    for ( int i2=0; i2 < N+1; ++i2 )
      for ( int e=0; e < num_of_el; ++e )
        h_in(i1,i2,e) = T( rand() ) / T( rand() );
 

  mode_vector<T,int> in(h_in), out(num_of_el, N+1); 


  volume0<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
    (N, in, out);

//  volume1<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
//    (N, in, out);

//  volume2<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
//    (N, in, out);

//  volume3<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
//    (N, in.data(), out.data());


  volume6<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
    (N, in, out);

  volume<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
    (N, in, out);


//  volume8<T><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
//    (N, in.data(), out.data());



  mode_matrix<T, int> d_m; 

#ifdef TEST_P
  if (N < 9)
#endif
#ifdef TEST_H
  if (num_of_el < 1024*1024 + 1)
#endif
  {
    host_laplacian_matrix<T,int> mm(num_of_el, N);
    d_m = mm;
  }
  else
  {
    host_laplacian_matrix<T,int> mm(1, N);
    d_m = mm;
  }


  volume_mvm<T,1><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
    (N, d_m, in, out);

#ifdef TEST_P
  if (N < 9)
#endif
#ifdef TEST_H
  if (num_of_el < 1024*1024 + 1)
#endif
    volume_mvm<T,0><<<dim3( (num_of_el + blockD - 1)/blockD , N+1, N+1), blockD>>>
      (N, d_m, in, out);



  cudaThreadSynchronize();

  error = cudaGetLastError();
  lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;

/*
  host_mode_vector<T, int> h_out(out);

  for (int i1 = 0; i1 < N+1; ++i1)
    for (int i2 = 0; i2 < N+1; ++i2)
      std::cout<<h_out(i1,i2,1)<<std::endl;
*/

  in.free();
  out.free();
  d_m.free();

  return 0;

}


int main(int argc, const char* argv[])
{
  int blockD = 256;

#ifdef TEST_P

  std::cerr<<"TEST_P: 2 4 8 16 32"<<std::endl;

  for(int i =2 ; i < 33; i *=2 ) 
    performance_test<float> (i, blockD, 512*512);

#endif


#ifdef TEST_H

  std::cerr<<"TEST_H: 128 256 512 1024 2048"<<std::endl;

  for(int i =128 ; i < 2049; i *=2 ) 
    performance_test<float> (4, blockD, i*i);

#endif

#if 0


  int num_of_warps;

  if ( argc > 1 )
   num_of_warps = atoi( argv[1] );
  else
  {
    num_of_warps = 2048*2048;
//    num_of_warps = 8192;
//  num_of_warps = 8192*16; // ictp
//    std::cerr<<"arg needed"<<std::endl;
//    return 1 ;
  }


  int N = 8;
  int num_of_el = 32*num_of_warps;

  performance_test<double> (N, blockD, num_of_el);

  std::cerr<<sizeof(double)*num_of_el*(N+1)<<" bytes"<<std::endl;


#endif


  cudaDeviceReset();


  return 0;

}

