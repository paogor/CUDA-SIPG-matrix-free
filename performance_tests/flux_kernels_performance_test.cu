#include<iostream>
#include<cstdlib>

#include"flux_kernels_optim.hpp"

#include<build_square_mesh.hpp>

#if 1
  #define TEST_P
#else
  #define TEST_H
#endif


template<typename T>
int test( int order, int dim )
{

  square_mesh<T> sq_mesh(dim);
  host_mode_vector<T,int> filler(dim*dim, order+1);

  for(int i = 0; i <= order; i++)
    for(int j = 0;  j<= order; j++)
      for(int idx = 0;  idx < dim*dim; idx++)
         filler(i,j,idx) = T(rand())/ T(rand());

  mode_vector<T,int> d_bb(filler);
  mode_vector<T,int> d_xx(filler);

  test_flux0 ( order, sq_mesh.device_info, d_bb, d_xx );
  test_flux8 ( order, sq_mesh.device_info, d_bb, d_xx );
  test_flux  ( order, sq_mesh.device_info, d_bb, d_xx );

  d_xx.free();
  d_bb.free();

  return 0;

}



int main()
{


#ifdef TEST_P
  std::cerr<<"TEST_P: 2 4 8 16 32"<<std::endl;

  for(int i =2 ; i < 9; i *=2 ) 
    test<double>( i, 512);
#endif


#ifdef TEST_H
  std::cerr<<"TEST_H: 128 256 512 1042 2048"<<std::endl;

  for(int i =128 ; i < 2049; i *=2 ) 
    test<double>( 4, i);
#endif


  return 0;

}


