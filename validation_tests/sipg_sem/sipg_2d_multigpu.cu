
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
#include<vector>

#include<mpi.h>
#include<halo_exchange.h>

#include<sipg_sem_2d_multigpu.hpp>
#include<sipg_sem_2d.hpp>

#define EXACT_SOLUTION_NO 2
#include"../analytical_solutions.hpp"


void print_tile(int n1, int n2, int mode, std::ofstream & file, host_mode_vector<double,int> & md)
{

  for(int m1 = 0; m1<mode; ++m1 )
    for(int m2 = 0; m2<mode; ++m2 )
    {
      for(int i = 0; i < n1; ++i)
      {
        for(int j = 0; j < n2; ++j)
          file<<" "<< md(m1, m2, n1*i + j);

        file<<std::endl;
      }
      file<<std::endl<<std::endl;
    }

  return;
}


int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);


  int pid, nprocs;
  MPI_Comm CartComm;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // output file 
  std::stringstream filename;
  filename<<"test"<<pid<<".txt";
  std::ofstream file(filename.str().c_str(), std::ofstream::out);

  std::cout << pid << " " << nprocs << std::endl;

  int dims[3] = {2, 2, 1};
  int period[3] = {0, 0, 0};

  MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

  int coords[3] = {0, 0, 0};
  MPI_Cart_get(CartComm, 3, dims, period, coords);

  file<<pid<<" coords ("<<coords[0]<<", "<<coords[1]<<", "<<coords[2]<<")"<<std::endl<<std::endl;


  const int dim = 7; // local dim without halos
  const int degree = 4;

  using namespace test_func;
  square_mesh_multigpu<double> sq_mesh( dim, 2, coords[0], coords[1] ); 
  sipg_sem_2d_multigpu<double> p(CartComm, degree, sq_mesh, f, u_ex, dx_u_ex, dy_u_ex);

//  print_tile( dim+2, dim+2, degree+1, file, p.solution );

  file<<p.L2_err<<"\t"<<p.H1_err<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

#if 0

  int dim_s = dim*2;

  host_mode_vector<double,int> h_input(dim_s*dim_s, degree+1);
 
  for (int i = 0; i< degree+1; ++i)
    for (int j = 0; j< degree+1; ++j)
      for (int e = 0; e< dim_s*dim_s; ++e)
         h_input(i,j,e) = double(i+1)/(j+1);
     
  host_mode_vector<double,int> d_input(h_input);

  mode_vector<double,int> d_output(dim_s*dim_s, degree+1);

  square_mesh<double> single_mesh(dim_s);
  sipg_sem_2d<double> single_p(degree, single_mesh, f, u_ex, dx_u_ex, dy_u_ex);

  single_p._mvm(single_p.d_rhs, d_output);

  host_mode_vector<double,int> h_output(d_output);

  file<<std::endl<<" +++++++++++++++++++++++++ "<<std::endl;

  print_tile( dim_s, dim_s, degree+1, file, h_output );

#endif

  return 0;

}
