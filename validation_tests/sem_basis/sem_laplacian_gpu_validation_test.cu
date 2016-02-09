#include<iostream>
#include<thrust/device_vector.h>
#include<thrust/host_vector.h>

#include<sem_gpu_kernels.hpp>

using thrust::device_vector;
using thrust::host_vector;
using thrust::raw_pointer_cast;


template<int N>
__global__ void test_deriv_sem_bases(double * output_Dphi, double * output_weights )
{

  int modes = threadIdx.x;
  int nodes = threadIdx.y;

  output_Dphi[modes*(N+1) + nodes] = Dphi<double>(N,modes,nodes);
  output_weights[nodes] = device_const::lgl_weights<double>()[nodes];

}


template<int N>
void test_Dphi ()
{

  load_Dphi_table<double>(N);
  load_lgl_quadrature_table<double>(N);

  device_vector<double> output_Dphi((N+1)*(N+1)), output_weights(N+1);
 

  cudaError_t error;
  std::string lastError;

  test_deriv_sem_bases<N><<<dim3(1,1,1),dim3(N+1,N+1,1)>>>
    (raw_pointer_cast(output_Dphi.data()), raw_pointer_cast(output_weights.data()));

  error = cudaGetLastError();
  lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;


  
  host_vector<double> h_output_Dphi = output_Dphi;
  host_vector<double> h_output_weights = output_weights;

  std::vector<double> dlp; 
  DbaseVALUES (N, dlp);

  for (int i = 0; i < dlp.size(); ++i)
    std::cout<<h_output_Dphi[i]<<"\t"<<dlp[i]<<"\t"<<h_output_Dphi[i]-dlp[i]<<std::endl;

  std::cout<<std::endl;

  LGL_quadrature_table<double> qt(N+1);

  for (int i = 0; i < output_weights.size(); ++i )
    std::cout<<h_output_weights[i]<<"\t"<<qt.weight(i)<<"\t"
             <<h_output_weights[i]-qt.weight(i)<<std::endl;


}

#include<laplacian_operator.hpp>

template<typename T>
void laplacian_2d(int N, T * a)
{

  laplace_2d<T> lap(N);

  int append = 0;

  for (int i1=0; i1 <= N; ++i1)
    for (int i2=0; i2 <= N; ++i2)
    {  
      for (int j1=0; j1 <= N; ++j1)
        for (int j2=0; j2 <= N; ++j2)
        {
          a[append] = lap(i1,i2,j1,j2);
     //    std::cerr<< a[append] <<"\t";
          ++append;
        }

    // std::cerr<<std::endl;
     }

}



template<typename T>
int laplacian_2d_GPU(int N, T * laplacian )
{

  cudaError_t error;
  std::string lastError;

  load_Dphi_table<T>(N);
  load_lgl_quadrature_table<T>(N);


  const int num_of_el = (N+1)*(N+1);
  host_mode_vector<T,int> h_in(num_of_el,N+1);


  for (int i1 = 0; i1 <= N; ++i1)
    for (int i2 = 0; i2 <= N; ++i2)
      h_in(i1, i2, ( i1*(N+1) + i2 ) ) = 1.;

  mode_vector<T,int> in(h_in), out(num_of_el,N+1);

//  mvm(N, in, out);


  volume<T><<<dim3(1, N+1, N+1),num_of_el>>> (N, in, out);

  error = cudaGetLastError();
  lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;


  host_mode_vector<T,int> h_out(out);

  cudaThreadSynchronize();



  int append = 0;

  for (int e = 0; e < num_of_el; ++e)
    for (int i1 = 0; i1 <= N; ++i1)
      for (int i2 = 0; i2 <= N; ++i2)
      {
        laplacian[append] = h_out(i1, i2, e);
        ++append;
      }


  in.free();
  out.free();

  return 0;

}

#include<cmath>
#include<algorithm>

#include<mode_matrix_kernels.hpp>


template<typename T>
int laplacian_2d_GPU_mode_matrix(int N, T * laplacian )
{

  cudaError_t error;
  std::string lastError;

  const int noe = (N+1)*(N+1);

  host_mode_matrix<T,int> m(noe, N);

  mode_matrix<double, int> d_m; 
  d_m = m;

  host_mode_vector<T,int> h_in(noe,N+1);

  for (int i1 = 0; i1 <= N; ++i1)
    for (int i2 = 0; i2 <= N; ++i2)
      h_in(i1, i2, ( i1*(N+1) + i2 ) ) = 1.;

  mode_vector<T,int> in(h_in), out(noe,N+1);

  volume_mvm<T>
  <<<dim3(noe, N+1, N+1), 1>>>
  (N, d_m, in, out);


  error = cudaGetLastError();
  lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;


  host_mode_vector<T,int> h_out(out);

  cudaThreadSynchronize();

  int append = 0;

  for (int e = 0; e < noe; ++e)
    for (int i1 = 0; i1 <= N; ++i1)
      for (int i2 = 0; i2 <= N; ++i2)
      {
        laplacian[append] = h_out(i1, i2, e);
        ++append;
      }


  in.free();
  out.free();

  return 0;

}


template<typename T>
T max_laplacian_error(int N)
{

  T * lapGPU = new T[(N+1)*(N+1)*(N+1)*(N+1)];
  T * lapCPU = new T[(N+1)*(N+1)*(N+1)*(N+1)];

  laplacian_2d_GPU_mode_matrix<T>(N, lapGPU);
  laplacian_2d<T>(N, lapCPU);

  T max_err = -1.;

  for(int i = 0; i < (N+1)*(N+1)*(N+1)*(N+1); ++i)
      max_err = std::max(std::fabs( lapGPU[i] - lapCPU[i] ), max_err);

  return max_err;

}


int main()
{

 test_Dphi<3>();

  for (int N = 2; N < 20; ++N )
    std::cout<<N<<": "<<max_laplacian_error<double>(N)<<std::endl;

  const int N = 3;
  double * lapGPU = new double[(N+1)*(N+1)*(N+1)*(N+1)];
  double * lapCPU = new double[(N+1)*(N+1)*(N+1)*(N+1)];


  laplacian_2d_GPU_mode_matrix<double>(N, lapGPU);
  laplacian_2d<double>(N, lapCPU);


  for(int i = 0; i < (N+1)*(N+1); ++i)
  {
    for(int j = 0; j < (N+1)*(N+1); ++j)
      std::cout<<lapGPU[i*(N+1)*(N+1) + j]<<"\t";
    std::cout<<std::endl;
  }

  std::cout<<std::endl;

  for(int i = 0; i < (N+1)*(N+1); ++i)
  {
    for(int j = 0; j < (N+1)*(N+1); ++j)
      std::cout<<lapCPU[i*(N+1)*(N+1) + j]<<"\t";
    std::cout<<std::endl;
  }

  return 0;
}

