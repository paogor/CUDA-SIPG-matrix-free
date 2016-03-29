#define __DEBUG_NO_MPI__

#include<iostream>
#include<dotproduct_multigpu.old2>
#include<build_square_mesh.hpp>



#ifndef __DEBUG_NO_MPI__
  #include<mpi.h>
#endif

int main(int argc, char** argv)
{

#ifndef __DEBUG_NO_MPI__
  MPI_Init(&argc, &argv);
#endif

  int local_dim = 1024;
  int noe = (local_dim+2)*(local_dim+2);
  int degree =4;

  square_mesh_multigpu<double> sq_mesh( local_dim, 4, 2, 2 ); 
  dotproduct_multigpu<double> test(sq_mesh.device_info, degree);

  host_mode_vector<double, int> h_a(noe, degree+1), h_b(noe, degree+1);

  for(int i1=0; i1<=degree; ++i1)
    for(int i2=0; i2<=degree; ++i2)
      for(int e=0; e < noe; ++e)
      {
        h_a(i1, i2, e) = .5*i1 + .34*i2 + e*4.4;
        h_b(i1, i2, e) = .15*i1 + .88*i2 + e*9.1;
      }

  mode_vector<double, int> a(h_a), b(h_b);

  double q = test(a,a);
  double w = test(b,b);
  double e = test(a,b);

  std::cout<<q<<"\t"<<w<<"\t"<<e<<std::endl; 

#ifndef __DEBUG_NO_MPI__
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif

  return 0;

}
