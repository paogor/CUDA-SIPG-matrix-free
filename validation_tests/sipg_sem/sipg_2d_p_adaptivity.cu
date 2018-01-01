/***
* /file 
*
* Validation test for Laplacian matrix.
* 
*   solve  - u" = f  with Dirichlet boundary conditions
*   Numerical solution is confronted with analytical one.
*
*/

#include<iostream>

#define EXACT_SOLUTION_NO 5
#include"../analytical_solutions.hpp"

//#define USE_MODE_MATRIX
#include<sipg_sem_2d.hpp>


#include<iomanip>

int main()
{

  std::cout<<"EXACT_SOLUTION_NO "<<EXACT_SOLUTION_NO<<std::endl;

#ifdef USE_MODE_MATRIX
  std::cout<<"USE_MODE_MATRIX is ON"<<std::endl;
#endif

  const int dim = 16;
  square_mesh<double> sq_mesh(dim);

  const double toll = 1e-14;

  std::cout<<"toll: "<<toll<<std::endl;

  for (int degree = 1; degree < 15; ++degree)
  { 

    const double penalty = degree*degree;

    using namespace test_func;
    sipg_sem_2d<double> p(degree, sq_mesh, f, u_ex, dx_u_ex, dy_u_ex, penalty, toll);

    std::cout<<dim<<"\t"<<degree<<"\t";
    std::cout<<std::setw(12)<<p.H1_err<<"\t";
    std::cout<<std::setw(12)<<p.L2_err<<"\t";
    std::cout<<std::setw(12)<<p.max_err;
//    std::cout<<"\t"<<it;
    std::cout<<std::endl;

  }

  sq_mesh.device_info.free();


#if 0
  cudaError_t error = cudaGetLastError();
  std::string lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;
#endif

  return 0;

}


