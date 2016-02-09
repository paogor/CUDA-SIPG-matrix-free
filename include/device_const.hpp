#ifndef __DEV_CONST_HPP__
#define __DEV_CONST_HPP__

#include<LGL_quadrature_table.hpp>
#include<sem_function.hpp>


// --------------------- begin DEVICE CONSTANTS ---------------------

#define LGL_CONST_ARRAY_LENGHT 64
__constant__ float  d_lgl_nodes_float    [LGL_CONST_ARRAY_LENGHT];
__constant__ float  d_lgl_weights_float  [LGL_CONST_ARRAY_LENGHT];
__constant__ double d_lgl_nodes_double   [LGL_CONST_ARRAY_LENGHT];
__constant__ double d_lgl_weights_double [LGL_CONST_ARRAY_LENGHT];

#define DPHI_CONST_ARRAY_LENGHT 4096
__constant__ double d_dphi_double [DPHI_CONST_ARRAY_LENGHT]; 
__constant__ float  d_dphi_float  [DPHI_CONST_ARRAY_LENGHT]; 

// ---------------------- end DEVICE CONSTANTS ----------------------


/**
  It contains functions for to manage and to access the device constant memory.
*/
namespace device_const
{

  template<typename T> __device__ inline T* lgl_nodes() { return NULL; }  // dumb

  template<>
  __device__ inline float* lgl_nodes<float>()
  { 
    return d_lgl_nodes_float;
  } 

  template<>
  __device__ inline double* lgl_nodes<double>()
  {
    return d_lgl_nodes_double;
  } 


  template<typename T> __device__ inline T* lgl_weights() { return NULL; }  // dumb

  template<>
  __device__ inline float* lgl_weights<float>()
  { 
    return d_lgl_weights_float;
  } 

  template<>
  __device__ inline double* lgl_weights<double>()
  {
    return d_lgl_weights_double;
  } 

}

template<typename T>
void load_lgl_quadrature_table( int order )
{
  return; // DUMB
}


template<>
void load_lgl_quadrature_table<float>( int order )
{

  LGL_quadrature_table<float> qt(order+1);

  checkError(cudaMemcpyToSymbol( d_lgl_nodes_float,
                                 qt.nodes_array(),
                                 sizeof(float)*(order+1) ));

  checkError(cudaMemcpyToSymbol( d_lgl_weights_float,
                                 qt.weights_array(),
                                 sizeof(float)*(order+1) ));
  return;
}

template<>
void load_lgl_quadrature_table<double>( int order )
{

  LGL_quadrature_table<double> qt(order+1);

  checkError(cudaMemcpyToSymbol( d_lgl_nodes_double,
                                 qt.nodes_array(),
                                 sizeof(double)*(order+1) ));

  checkError(cudaMemcpyToSymbol( d_lgl_weights_double,
                                 qt.weights_array(),
                                 sizeof(double)*(order+1) ));
  return;
}



template<typename T>
void DbaseVALUES (int N, std::vector<T> & values)
{

  sem_function<T> b(N);

  values.resize((N+1)*(N+1));

  for (int j = 0; j < N+1; ++j)
    for (int z = 0; z < N+1; ++z)
      values[j*(N+1) + z] = b.d_phi(j,z);

}


// ----------------------

template<typename T>
void load_Dphi_table(int order)
{
  return;
}



template<>
void load_Dphi_table<double>( int order )
{
  std::vector<double> dlp;

  DbaseVALUES (order, dlp);

  checkError(cudaMemcpyToSymbol( d_dphi_double,
                                 dlp.data(),
                                 sizeof(double)*dlp.size() ));

  return;
}



template<>
void load_Dphi_table<float>( int order )
{
  std::vector<float> dlp;

  DbaseVALUES (order, dlp);

  checkError(cudaMemcpyToSymbol( d_dphi_float,
                                 dlp.data(),
                                 sizeof(float)*dlp.size() ));

  return;
}




//// WARNING Dphi need to be templatized



template<typename FLOAT_TYPE>
inline __device__ FLOAT_TYPE Dphi(int N, int mode, int node)
{
  return NULL; //dump
}


template<>
inline __device__ double Dphi<double>(int N, int mode, int node)
{
   return d_dphi_double[ mode*(N+1) + node ]; 
}


template<>
inline __device__ float Dphi<float>(int N, int mode, int node)
{
   return d_dphi_float[ mode*(N+1) + node ]; 
}


// ----------------------


template<typename FLOAT_TYPE>
inline __device__ FLOAT_TYPE * DPHI(int N, int mode)
{
  return NULL; //dump
}


template<>
inline __device__ double * DPHI<double>(int N, int mode)
{
   return d_dphi_double + (mode*(N+1));
}


template<>
inline __device__ float * DPHI<float>(int N, int mode)
{
   return d_dphi_float + (mode*(N+1));
}


// ----------------------


/*
inline __device__ double Dphi(int N, int mode, int node)
{
   return d_dphi_double[ mode*(N+1) + node ]; 
}


inline __device__ double * DPHI(int N, int mode)
{
   return d_dphi_double + (mode*(N+1));
}
*/


#endif 
