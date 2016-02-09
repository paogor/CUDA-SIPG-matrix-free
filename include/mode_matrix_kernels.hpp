#ifndef __MODE_MATRIX_KERNELS_HPP__
#define __MODE_MATRIX_KERNELS_HPP__

#include<mode_matrix.hpp>
#include<mode_vector.hpp>

template<typename T, bool local_operators_are_all_the_same>
__global__ void volume_mvm ( int N,
                             mode_matrix<T, int> mat,
                             mode_vector<T, int> input,
                             mode_vector<T, int> output )
{

  int idx = blockIdx.x*blockDim.x + threadIdx.x; 

  int i1 = blockIdx.y;
  int i2 = blockIdx.z;

  if ( !(idx < input.get_noe()) ) return;

  //  i1,i2 output modes

  T * row(NULL);

  if (local_operators_are_all_the_same)
  row = mat(i1,i2,0);
  else row = mat(i1,i2,idx);


  const int next = mat.jump_to_next();
  T result = 0.;

  // It takes i1,i2 as INPUT. 

  result += *row * input(i1,i2,idx);


  // (i1,[0..N]/i2;i1,i2) extra-diagonal term: Laplacian
  // It takes i1,[0..N] as INPUT.


  for (int i=0; i <= N; ++i)
    if ( i != i2 )
    {
      row += next; 
      result +=  *row * input(i1,i,idx);
    }


  // ([0..N]/i1,i2;i1,i2) extra-diagonal term: Laplacian
  // It takes [0..N],i2 as INPUT.


  for (int i=0; i <= N; ++i)
    if ( i != i1 )
    {
      row += next; 
      result +=  *row * input(i,i2,idx);
    }


   output(i1,i2,idx) = result;
   return;

}

#endif

