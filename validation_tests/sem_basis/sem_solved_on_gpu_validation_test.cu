/***
* /file 
*
* Validation test for Laplacian matrix.
* 
*   solve  - u" = f  with homogeneous Dirichlet border conditions
*   Numerical solution is confronted with analytical one.
*
*/

#include<iostream>
#include<algorithm>
#include<cmath>

#include<cublas_wrapper.hpp>

#include<conjugate_gradient_gpu.hpp>
#include<mode_vector.hpp>
#include<sem_gpu_kernels.hpp>

#include<LGL_quadrature_table.hpp>
#include<sem_function.hpp>
















template<typename FLOAT_TYPE>
void copy_and_show (mode_vector<FLOAT_TYPE, int> target)
{
  host_mode_vector<FLOAT_TYPE, int> h_target(target);

  int e = 0;

  for(int i1 = 0; i1 < h_target.get_nompe(); ++i1)
    for(int i2 = 0; i2 < h_target.get_nompe(); ++i2)
    {
     // for(int e = 0; e < h_target.get_noe(); ++e)
        std::cerr<<h_target(i1,i2,e)<<"\t";
      std::cerr<<std::endl;
    }
}


template<typename T>
T u_ex (T x, T y)
{
  const T PI = 3.14159265359;
  return x*y*(y-1)*(x-1)*std::exp(x-y);   // A
//  return x*(1-x)*std::exp(3*x)*std::sin(2*PI*y);  // B
//  return y*(1-y)*x*x*x; // C

//  return x*x*y*y; //D

//  return std::cos(8*PI*x) + std::cos(8*PI*y); //E

}


template<typename T>
T f (T x, T y)
{
  const T PI = 3.14159265359;
  return -2*x*(y-1)*(y-2*x+x*y+2)*std::exp(x-y); // A
//  return -std::sin(2*PI*y)*std::exp(3*x)*((4-3*x-9*x*x)-4*PI*PI*(x-x*x)); // B
//  return 2*x*x*x - 6*x*y*(1-y); // C 

//  return -2*y*y -2*x*x; //D

//  return 64*PI*PI*std::cos(8*PI*x) + 64*PI*PI*std::cos(8*PI*y); //E

//  return 1.;
}




int vec_INDEX(int N, int i1, int i2)
{
  return ((N+1)*i2 + i1) ;
}





template<typename T>
void bc_nitsche_method_rhs (int N, std::vector<T> & rhs)
{

  sem_function<T> b(N);

  std::vector<T> nodes(N+1), weights(N+1);

  LGL_quadrature_table<T>  qt(N+1);

  for (int i = 0; i <= N; ++i)
  {
    nodes[i] = qt.node(i)*.5 + .5;
    weights[i] = qt.weight(i)*.5;
  }

  for (int i=0; i<= N; ++i)
  for (int j=0; j<= N; ++j)
  {
    //DOWN
    rhs[ vec_INDEX(N,i,j) ] -= -1*weights[i]*b.d_phi(j,0)*u_ex(nodes[i],nodes[0])*2;

    //UP
    rhs[ vec_INDEX(N,i,j) ] -=    weights[i]*b.d_phi(j,N)*u_ex(nodes[i],nodes[N])*2;

    //LEFT
    rhs[ vec_INDEX(N,j,i) ] -= -1*weights[i]*b.d_phi(j,0)*u_ex(nodes[0],nodes[i])*2;

    //RIGHT
    rhs[ vec_INDEX(N,j,i) ] -=    weights[i]*b.d_phi(j,N)*u_ex(nodes[N],nodes[i])*2;
  } 

  const T pen = 10; // check
  for (int i=0; i <= N; ++i)
  {
    rhs[ vec_INDEX(N,N,i) ] += pen*weights[i]*u_ex(nodes[N],nodes[i]);
    rhs[ vec_INDEX(N,i,N) ] += pen*weights[i]*u_ex(nodes[i],nodes[N]);

    rhs[ vec_INDEX(N,0,i) ] += pen*weights[i]*u_ex(nodes[0],nodes[i]);
    rhs[ vec_INDEX(N,i,0) ] += pen*weights[i]*u_ex(nodes[i],nodes[0]);
  }


}




#include<iomanip>

template<typename T>
int poisson_2d_homogeneous_dirichlet_bc( int order,
                                       //  std::vector<T> & solution,
                                         T & L2_err,
                                         T & max_err,
                                         T & index_max_err )
{

  const int noe = 1;

  // ********************************** b **********************************

  host_mode_vector<T,int> d_bb(noe, order+1);

  std::vector<T> nodes(order+1), weights(order+1);
  LGL_quadrature_table<T> qt(order+1);

  const int vector_size = (order+1)*(order+1);
  std::vector<T> b(vector_size, 0); //< rhs vector

  for (int i = 0; i <= order; ++i)
  {
    nodes[i] = qt.node(i)*.5 + .5;
    weights[i] = qt.weight(i)*.5;
  }

  for (int i1 = 0; i1 <= order; ++i1)
    for (int i2 = 0; i2 <= order; ++i2)
      b[vec_INDEX(order,i1,i2)] = weights[i1]*weights[i2]*f(nodes[i1], nodes[i2]);

  bc_nitsche_method_rhs (order, b);

  for (int i1 = 0; i1 <= order; ++i1)
    for (int i2 = 0; i2 <= order; ++i2)
      for (int e = 0; e < noe; ++e)
        d_bb(i1,i2,e) = b[vec_INDEX(order,i1,i2)]; 


  // ****************************** solve Ax=b *****************************

  host_mode_vector<T,int> d_xx(noe, order+1);

  mode_vector<T,int> xx(d_xx);
  mode_vector<T,int> bb(d_bb);
//copy_and_show(bb);


  sem_2d_test<T> problem(order);

  // run function to solve
  int it = conjugate_gradient(problem, xx, bb );

  // copy back the solution 
  copy(xx, d_xx);

  std::vector<T> err;

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      err.push_back ( std::fabs(d_xx(i,j,0) - u_ex(nodes[i], nodes[j])));

  typename std::vector<T>::iterator m = std::max_element(err.begin() , err.end() );

  L2_err = 0;
  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      L2_err +=  weights[i]*weights[j]*std::pow(d_xx(i,j,0) - u_ex(nodes[i], nodes[j]), 2);


  L2_err = std::sqrt(L2_err);

  max_err = *m;

  index_max_err = m - err.begin(); 

#if 1

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      std::cerr<<std::setw(10)<<nodes[i]<<"\t"<<std::setw(10)<<nodes[j]<<"\t"
               <<std::setw(10)<<d_xx(i,j,0)<<"\t"<<std::setw(10)
               <<u_ex(nodes[i],nodes[j])<<std::endl;
  
#endif

  xx.free();
  bb.free();

  return it;
}










int main()
{

//  int i = 3;

  for (int i = 0; i < 64; ++i)
  { 
    double L2_err, max_err, index_max_err; 
    int it = poisson_2d_homogeneous_dirichlet_bc(i, L2_err, max_err, index_max_err);

    std::cout<<i<<":\t"<<L2_err<<",\t"<<max_err<<",\t "<<index_max_err<<",\t "<<it<<std::endl;
  }

  return 0;
}



