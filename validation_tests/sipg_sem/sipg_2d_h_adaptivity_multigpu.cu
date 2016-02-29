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
#include<fstream>
#include<sstream>
#include<string>
#include<vector>

#include<mpi.h>

#define MPI_NODE_PER_EDGE 4

#define EXACT_SOLUTION_NO 1
#include"../analytical_solutions.hpp"

#include<sipg_sem_2d_multigpu.hpp>

#include<iomanip>

#include<CUDA_TIMER.hpp>

int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  int pid, nprocs;
  MPI_Comm CartComm;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if(pid == 0)
  {
    std::cerr<<"EXACT_SOLUTION_NO "<<EXACT_SOLUTION_NO<<std::endl;
    std::cerr<<"MPI_NODE_PER_EDGE "<<MPI_NODE_PER_EDGE<<std::endl;
  }

  int dims[3] = {MPI_NODE_PER_EDGE, MPI_NODE_PER_EDGE, 1};
  int period[3] = {0, 0, 0};

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

  int coords[3] = {0, 0, 0};
  MPI_Cart_get(CartComm, 3, dims, period, coords);


  int degree = 5;

//  for (int degree = 2; degree < 6; ++degree)
  {
  double L2_err_old(0), H1_err_old(0);

  for (int dim = 4; dim < 259; dim*=2)
  {

    CUDA_TIMER t;
    using namespace test_func;
    if (pid == 0) t.start();
    square_mesh_multigpu<double> sq_mesh( dim, MPI_NODE_PER_EDGE, coords[0], coords[1] ); 
    sipg_sem_2d_multigpu<double> p(CartComm, degree, sq_mesh, f, u_ex, dx_u_ex, dy_u_ex);
    if (pid == 0) t.stop();


    if(pid == 0)
    {

      std::cerr<<MPI_NODE_PER_EDGE*dim<<"\t"<<degree<<"\t";
      std::cerr<<std::setw(12)<<log(p.H1_err/H1_err_old)/log(2)<<"\t";
      std::cerr<<std::setw(12)<<p.H1_err<<"\t";
      std::cerr<<std::setw(12)<<log(p.L2_err/L2_err_old)/log(2)<<"\t";
      std::cerr<<std::setw(12)<<p.L2_err<<"\t";
      std::cerr<<t.elapsed_millisecs();
      std::cerr<<std::endl;

      L2_err_old = p.L2_err;
      H1_err_old = p.H1_err;

    }

    sq_mesh.device_info.free();

  }


  if (pid == 0) std::cerr<<std::endl;
}



#if 0
  cudaError_t error = cudaGetLastError();
  std::string lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;
#endif

  MPI_Finalize();

  return 0;

}


