#include<iostream>
#include<mode_vector.hpp>


template<typename T>
T u_ex (T x, T y)
{
  const T PI = 3.14159265359;
//  return x*y*(y-1)*(x-1)*std::exp(x-y);   // A
//  return x*(1-x)*std::exp(3*x)*std::sin(2*PI*y);  // B
//  return y*(1-y)*x*x*x; // C

//  return x*x*y*y; //D

  return std::cos(8*PI*x) + std::cos(8*PI*y); //E

}



template<typename T>
T f (T x, T y)
{
  const T PI = 3.14159265359;
//  return -2*x*(y-1)*(y-2*x+x*y+2)*std::exp(x-y); // A
//  return -std::sin(2*PI*y)*std::exp(3*x)*((4-3*x-9*x*x)-4*PI*PI*(x-x*x)); // B
//  return 2*x*x*x - 6*x*y*(1-y); // C 

//  return -2*y*y -2*x*x; //D

  return 64*PI*PI*std::cos(8*PI*x) + 64*PI*PI*std::cos(8*PI*y); //E

//  return 1.;
}


#include<rhs.hpp>

int main()
{

  const int N = 3;
  const int half_dim = 3;
  const int dim = 2*half_dim;

  // +++++++++++

  square_mesh<double> mesh(dim);

  host_mode_vector<double,int> rhs(dim*dim, N+1);

  bc_nitsche_method_rhs<double> ( N, mesh, rhs);

  // +++++++++++

  square_mesh_multigpu<double> multi_mesh( dim, 1, 0, 0 ); 

  host_mode_vector<double,int> multi_rhs( (dim+2)*(dim+2), N+1);

  bc_nitsche_method_rhs<double> ( N, multi_mesh, multi_rhs);

  // +++++++++++

#if 0 
//  for(int j1 = 0; j1 < N+1; ++j1)
//    for(int j2 = 0; j2 < N+1; ++j2)
    {

      int j1 = 0; int j2 = 0;

      for(int y = 0; y <dim; y++) 
      {
        for(int x = 0; x <dim; x++) 
        {
          double err =         rhs(j1, j2,     y*dim     + x )
                       - multi_rhs(j1, j2, (y+1)*(dim+2) + (x+1) ); 

 //         std::cout<<" "<<err;

          std::cout<<"   "<<    mesh.x_coord()[ y*dim     + x ] <<" | "<<
                    multi_mesh.x_coord()[(y+1)*(dim+2) + (x+1) ]; 

        }

        std::cout<<std::endl;

      }

      std::cout<<std::endl;

    }
#endif

  // +++++++++++

  square_mesh_multigpu<double> multi_mesh0( half_dim, 2, 0, 0 ); 
  square_mesh_multigpu<double> multi_mesh1( half_dim, 2, 1, 0 ); 
  square_mesh_multigpu<double> multi_mesh2( half_dim, 2, 0, 1 ); 
  square_mesh_multigpu<double> multi_mesh3( half_dim, 2, 1, 1 ); 

  host_mode_vector<double,int> multi_rhs0( (half_dim+2)*(half_dim+2), N+1);
  host_mode_vector<double,int> multi_rhs1( (half_dim+2)*(half_dim+2), N+1);
  host_mode_vector<double,int> multi_rhs2( (half_dim+2)*(half_dim+2), N+1);
  host_mode_vector<double,int> multi_rhs3( (half_dim+2)*(half_dim+2), N+1);


#if 0
  std::cout<<multi_mesh0.DOWNborder()<<" "<<multi_mesh0.RIGHTborder()<<" ";
  std::cout<<multi_mesh0.UPborder()<<" "<<multi_mesh0.LEFTborder()<<std::endl;

  std::cout<<multi_mesh1.DOWNborder()<<" "<<multi_mesh1.RIGHTborder()<<" ";
  std::cout<<multi_mesh1.UPborder()<<" "<<multi_mesh1.LEFTborder()<<std::endl;

  std::cout<<multi_mesh2.DOWNborder()<<" "<<multi_mesh2.RIGHTborder()<<" ";
  std::cout<<multi_mesh2.UPborder()<<" "<<multi_mesh2.LEFTborder()<<std::endl;

  std::cout<<multi_mesh3.DOWNborder()<<" "<<multi_mesh3.RIGHTborder()<<" ";
  std::cout<<multi_mesh3.UPborder()<<" "<<multi_mesh3.LEFTborder()<<std::endl;
#endif


  bc_nitsche_method_rhs<double> ( N, multi_mesh0, multi_rhs0);
  bc_nitsche_method_rhs<double> ( N, multi_mesh1, multi_rhs1);
  bc_nitsche_method_rhs<double> ( N, multi_mesh2, multi_rhs2);
  bc_nitsche_method_rhs<double> ( N, multi_mesh3, multi_rhs3);

  // +++++++++++

#if 1 
  for(int j1 = 0; j1 < N+1; ++j1)
    for(int j2 = 0; j2 < N+1; ++j2)
    {

      for(int y = 0; y <half_dim; y++) 
      {

        for(int x = 0; x <half_dim; x++) 
          std::cout<<"   "<<    rhs(j1, j2,     y*dim     + x ) 
                   - multi_rhs0(j1, j2, (y+1)*(half_dim+2) + (x+1) ); 

        std::cout<<" |";

        for(int x = 0; x <half_dim; x++) 
          std::cout<<"   "<<    rhs(j1, j2,     y*dim     + x + half_dim) 
                   - multi_rhs1(j1, j2, (y+1)*(half_dim+2) + (x+1) ); 

        std::cout<<std::endl;

      }

      std::cout<<"-----------------------"<<std::endl;

      for(int y = 0; y <half_dim; y++) 
      {

        for(int x = 0; x <half_dim; x++) 
          std::cout<<"   "<<    rhs(j1, j2,     (y+half_dim)*dim     + x ) 
                   - multi_rhs2(j1, j2, (y+1)*(half_dim+2) + (x+1) ); 

        std::cout<<" |";

        for(int x = 0; x <half_dim; x++) 
          std::cout<<"   "<<    rhs(j1, j2,    (y+half_dim)*dim     + x + half_dim) 
                   - multi_rhs3(j1, j2, (y+1)*(half_dim+2) + (x+1) ); 

        std::cout<<std::endl;

      }

      std::cout<<std::endl;


    }
#endif

#if 1
  std::cout<<"mesh.dim(): "<<mesh.dim()<<std::endl;
  std::cout<<"multi_mesh3.dim(): "<<multi_mesh0.dim()<<std::endl;
#endif

  return 0;

} 


