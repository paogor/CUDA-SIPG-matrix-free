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

#define USE_MODE_MATRIX
#include<sipg_sem_2d.hpp>

#include<iomanip>
#include<CUDA_TIMER.hpp>

int main()
{

  std::cout<<"EXACT_SOLUTION_NO "<<EXACT_SOLUTION_NO<<std::endl;

#ifdef USE_MODE_MATRIX
  std::cout<<"USE_MODE_MATRIX is ON"<<std::endl;
#endif

  int degree = 4;

//  for (int degree = 2; degree < 4; ++degree)
  {
  double L2_err_old(0), H1_err_old(0);
  

  for (int dim = 2048; dim < 2049; dim*=2)
  {

    CUDA_TIMER t;
    using namespace test_func;
    t.start();
    square_mesh<double> sq_mesh(dim);
    sipg_sem_2d<double> p(degree, sq_mesh, f, u_ex, dx_u_ex, dy_u_ex);
    t.stop();

    std::cout<<dim<<"\t"<<degree<<"\t";
    std::cout<<std::setw(12)<<log(p.H1_err/H1_err_old)/log(2)<<"\t";
    std::cout<<std::setw(12)<<p.H1_err<<"\t";
    std::cout<<std::setw(12)<<log(p.L2_err/L2_err_old)/log(2)<<"\t";
    std::cout<<std::setw(12)<<p.L2_err<<"\t";
    std::cout<<std::setw(12)<<p.max_err;
//    std::cout<<"\t"<<it;
    std::cout<<"\t"<<t.elapsed_millisecs();
    std::cout<<std::endl;


    L2_err_old = p.L2_err;
    H1_err_old = p.H1_err;


    sq_mesh.device_info.free();

  }
    std::cout<<std::endl;
}



#if 0
  cudaError_t error = cudaGetLastError();
  std::string lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;
#endif

  return 0;

}


