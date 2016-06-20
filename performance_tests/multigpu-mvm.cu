#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>

#include<mpi.h>

#define __MVM_MULTIGPU_TEST__
#define MPI_NODE_PER_EDGE 4 

#define EXACT_SOLUTION_NO 5 
#include<../validation_tests/analytical_solutions.hpp>

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
#ifdef USE_MODE_MATRIX
    std::cout<<"USE_MODE_MATRIX is ON"<<std::endl;
#endif
  }

  int dims[3] = {MPI_NODE_PER_EDGE, MPI_NODE_PER_EDGE, 1};
  int period[3] = {0, 0, 0};

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

  int coords[3] = {0, 0, 0};
  MPI_Cart_get(CartComm, 3, dims, period, coords);


 int degree = 8;
//  for (int degree = 4; degree < 9; degree*=2)
  {

//    const int dim = 1024;
    for (int dim = 128; dim < 1025; dim*=2)
    {

      CUDA_TIMER t;
      using namespace test_func;
      square_mesh_multigpu<double> sq_mesh( dim, MPI_NODE_PER_EDGE, coords[0], coords[1] ); 
      sipg_sem_2d_multigpu<double> p(CartComm, degree, sq_mesh, f, u_ex, dx_u_ex, dy_u_ex, 1e-15);

      p._mvm ( p.d_rhs );
      t.start();
      for(int t=0; t < 20; ++t)
      {
        p._mvm ( p.d_rhs );
        p._mvm ( p.d_rhs );
        p._mvm ( p.d_rhs );
        p._mvm ( p.d_rhs );
        p._mvm ( p.d_rhs );
      }
      t.stop();

      float mean_time(0);
      float local_time = t.elapsed_millisecs()/100.0;

      MPI_Allreduce(&local_time, &mean_time, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

      mean_time = mean_time / (MPI_NODE_PER_EDGE*MPI_NODE_PER_EDGE);

      if(pid == 0)
      {

        std::cerr<<MPI_NODE_PER_EDGE*MPI_NODE_PER_EDGE<<"\t";
        std::cerr<<MPI_NODE_PER_EDGE*dim<<"\t"<<degree<<"\t";
        std::cerr<<dim<<"\t";
        std::cerr<<mean_time;
        std::cerr<<std::endl;

      }

      sq_mesh.device_info.free();

      MPI_Barrier(MPI_COMM_WORLD);

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


