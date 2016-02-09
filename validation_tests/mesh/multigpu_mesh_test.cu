#include<iostream>
#include<build_square_mesh.hpp>

int main()
{

  square_mesh_multigpu<double> m( 5, 10, 5, 5 ); 

   std::cout<<"dim(): "<<m.dim()<<std::endl;
   std::cout<<"local_dim(): "<<m.local_dim()<<std::endl;


   std::cout<<"device_info.get_dimx(): "<<m.device_info.get_dimx()<<std::endl;
   std::cout<<"device_info.get_dimy(): "<<m.device_info.get_dimy()<<std::endl;

   std::cout<<"device_info.compute_idx(x,y): "<<std::endl;

   for(int y = 0; y < m.device_info.get_dimy(); ++y)
   {
     for(int x = 0; x < m.device_info.get_dimx(); ++x)
       std::cout<<"  "<<m.device_info.compute_idx(x,y);

     std::cout<<std::endl;
   }

   std::cout<<"device_info.get_neighborhood_UP(x,y): "<<std::endl;

   for(int y = 0; y < m.device_info.get_dimy(); ++y)
   {
     for(int x = 0; x < m.device_info.get_dimx(); ++x)
       std::cout<<"  "<<m.device_info.get_neighborhood_UP(x,y);

     std::cout<<std::endl;
   }


   std::cout<<"device_info.get_x(0,e): "<<std::endl;

   for(int y = 0; y < m.device_info.get_dimy(); ++y)
   {
     for(int x = 0; x < m.device_info.get_dimx(); ++x)
       std::cout<<"  "<<m.get_x( 0, m.device_info.compute_idx(x,y) );

     std::cout<<std::endl;
   }



   std::cout<<" +++++++++++++++++++++++++++++++++++++++++ "<<std::endl;


  square_mesh<double> mm(5); 
   std::cout<<"device_info.compute_idx(x,y): "<<std::endl;

   for(int y = 0; y < mm.device_info.get_dimy(); ++y)
   {
     for(int x = 0; x < mm.device_info.get_dimx(); ++x)
       std::cout<<"  "<<mm.device_info.compute_idx(x,y);

     std::cout<<std::endl;
   }

   std::cout<<"device_info.get_neighborhood_UP(x,y): "<<std::endl;

   for(int y = 0; y < mm.device_info.get_dimy(); ++y)
   {
     for(int x = 0; x < mm.device_info.get_dimx(); ++x)
       std::cout<<"  "<<mm.device_info.get_neighborhood_UP(x,y);

     std::cout<<std::endl;
   }


   std::cout<<std::endl<<" +++++++++++++++++++++++++++++++++++++++++ "<<std::endl;



  square_mesh_multigpu<double> m0( 5, 3, 0, 0 ); 
  square_mesh_multigpu<double> m1( 5, 3, 1, 1 ); 
  square_mesh_multigpu<double> m2( 5, 3, 2, 2 ); 

  for(int x = 0; x < m0.device_info.get_dimx(); ++x)
    std::cout<<"  "<<m0.get_x( 0, m0.device_info.compute_idx(x,0) );

  std::cout<<" | ";

  for(int x = 0; x < m1.device_info.get_dimx(); ++x)
    std::cout<<"  "<<m1.get_x( 0, m1.device_info.compute_idx(x,0) );

  std::cout<<" | ";

  for(int x = 0; x < m2.device_info.get_dimx(); ++x)
    std::cout<<"  "<<m2.get_x( 0, m2.device_info.compute_idx(x,0) );


  std::cout<<std::endl;

  for(int y = 0; y < m0.device_info.get_dimy(); ++y)
    std::cout<<"  "<<m0.get_y( 0, m0.device_info.compute_idx(0,y) );

  std::cout<<" | ";

  for(int y = 0; y < m1.device_info.get_dimy(); ++y)
    std::cout<<"  "<<m1.get_y( 0, m1.device_info.compute_idx(0,y) );

  std::cout<<" | ";

  for(int y = 0; y < m2.device_info.get_dimy(); ++y)
    std::cout<<"  "<<m2.get_y( 0, m2.device_info.compute_idx(0,y) );


  std::cout<<std::endl;

  return 0;

} 

