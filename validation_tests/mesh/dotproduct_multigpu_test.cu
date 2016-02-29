#include<iostream>
#include<dotproduct_multigpu.hpp>
#include<build_square_mesh.hpp>
#include<mpi.h>


int main(int argc, char** argv)
{

  MPI_Init(&argc, &argv);

  square_mesh_multigpu<double> sq_mesh( 18, 4, 2, 2 ); 
  dotproduct_multigpu<double> test(sq_mesh.device_info, 5);

  host_mode_vector<double, int> h_a(20*20, 6);

  for(int i1=0; i1<6; ++i1)
    for(int i2=0; i2<6; ++i2)
      for(int e=0; e < 20*20; ++e)
        h_a(i1, i2, e) = 1.;


  mode_vector<double, int> a(h_a);

  std::cout<<test(a,a)<<std::endl;

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return 0;

}
