
#include<vector>

#include<build_square_mesh.hpp>
#include<sem_function.hpp>
#include<LGL_quadrature_table.hpp>



/** **DEPRECATED** This function has been rewrote as `sipg_2d` method */


template<typename T>
void bc_nitsche_method_rhs ( int N, const square_mesh<T> & mesh, 
                             host_mode_vector<T,int> & rhs)
{

  const int dim = mesh.dim();
  const int noe = mesh.noe();

  std::vector<T> x = mesh.x_coord();
  std::vector<T> y = mesh.y_coord();


  sem_function<T> b(N);

  const T pen = 100*N*N;

  std::vector<T> nodes(N+1), weights(N+1), w(N+1);
  LGL_quadrature_table<T> qt(N+1);

  for (int i = 0; i <= N; ++i)
  {
    nodes[i] = (qt.node(i)*.5 + .5)/dim;
    weights[i] = qt.weight(i);
    w[i] = qt.weight(i)*.5/dim;
  }

  for (int i1 = 0; i1 <= N; ++i1)
    for (int i2 = 0; i2 <= N; ++i2)
      for (int e = 0; e < noe; ++e)
        rhs(i1,i2,e) = w[i1]*w[i2]
                         *f( x[e] + nodes[i1],
                             y[e] + nodes[i2] );
  for (int e = 0; e < noe; e++)
  {


    if ( mesh.get_neighbor(0,e) == -1 ) //DOWN
      for (int i=0; i<= N; ++i)
      {
        rhs(i,0,e)  += pen*weights[i]*u_ex(x[e] + nodes[i], y[e] + nodes[0]);
        for (int j=0; j<= N; ++j)
          rhs(i,j,e) -= -1*weights[i]*b.d_phi(j,0)*u_ex(x[e] + nodes[i], y[e] + nodes[0]);
      }


    if ( mesh.get_neighbor(2,e) == -3 ) //UP
      for (int i=0; i<= N; ++i)
      {
        rhs(i,N,e)  += pen*weights[i]*u_ex(x[e] + nodes[i], y[e] + nodes[N]);
        for (int j=0; j<= N; ++j)
          rhs(i,j,e) -= weights[i]*b.d_phi(j,N)*u_ex(x[e] + nodes[i], y[e] + nodes[N]);
       }


    if ( mesh.get_neighbor(3,e) == -4 ) //LEFT
      for (int i=0; i<= N; ++i)
      {
        rhs(0,i,e)  += pen*weights[i]*u_ex(x[e] + nodes[0], y[e] + nodes[i]);
        for (int j=0; j<= N; ++j)
          rhs(j,i,e) -= -1*weights[i]*b.d_phi(j,0)*u_ex(x[e] + nodes[0], y[e] + nodes[i]);
      }


    if ( mesh.get_neighbor(1,e) == -2 ) //RIGHT
      for (int i=0; i<= N; ++i)
      {
        rhs(N,i,e)  += pen*weights[i]*u_ex(x[e] + nodes[N], y[e] + nodes[i]);
        for (int j=0; j<= N; ++j)
          rhs(j,i,e) -=    weights[i]*b.d_phi(j,N)*u_ex(x[e] + nodes[N], y[e] + nodes[i]);
      } 


  }

}





template<typename T>
void bc_nitsche_method_rhs ( int N, const square_mesh_multigpu<T> & mesh, 
                             host_mode_vector<T,int> & rhs)
{

  const int dim = mesh.dim();
  const int noe = mesh.noe();
  const int local_dim = mesh.local_dim();

  std::vector<T> x = mesh.x_coord();
  std::vector<T> y = mesh.y_coord();


  sem_function<T> b(N);

  const T pen = 100*N*N;

  std::vector<T> nodes(N+1), weights(N+1), w(N+1);
  LGL_quadrature_table<T> qt(N+1);

  for (int i = 0; i <= N; ++i)
  {
    nodes[i] = (qt.node(i)*.5 + .5)/dim;
    weights[i] = qt.weight(i);
    w[i] = qt.weight(i)*.5/dim;
  }

  for (int i1 = 0; i1 <= N; ++i1)
    for (int i2 = 0; i2 <= N; ++i2)
      for (int e = 0; e < noe; ++e)
        rhs(i1,i2,e) = w[i1]*w[i2]
                         *f( x[e] + nodes[i1],
                             y[e] + nodes[i2] );



  if( mesh.DOWNborder() )  
    for ( int xx = 0; xx < local_dim; ++xx ) //DOWN
    {
 
      int e = (local_dim - 2) * local_dim + xx;

      for (int i=0; i<= N; ++i)
      {
        rhs(i,0,e)  += pen*weights[i]*u_ex( x[e] + nodes[i], y[e] + nodes[0]);
        for (int j=0; j<= N; ++j)
          rhs(i,j,e) -= -1*weights[i]*b.d_phi(j,0)*u_ex(x[e] + nodes[i], y[e] + nodes[0]);
      }
    }



  if( mesh.UPborder() ) 
    for ( int xx = 0; xx < local_dim; ++xx ) //UP
    {

      int e = local_dim + xx;

      for (int i=0; i<= N; ++i)
      {
        rhs(i,N,e)  += pen*weights[i]*u_ex(x[e] + nodes[i], y[e] + nodes[N]);
        for (int j=0; j<= N; ++j)
          rhs(i,j,e) -= weights[i]*b.d_phi(j,N)*u_ex(x[e] + nodes[i], y[e] + nodes[N]);
       }
    }




  if ( mesh.LEFTborder() ) 
    for ( int yy = 0; yy < local_dim; ++yy ) //LEFT
    {

      int e = yy*local_dim + 1;

      for (int i=0; i<= N; ++i)
      {
        rhs(0,i,e)  += pen*weights[i]*u_ex(x[e] + nodes[0], y[e] + nodes[i]);
        for (int j=0; j<= N; ++j)
          rhs(j,i,e) -= -1*weights[i]*b.d_phi(j,0)*u_ex(x[e] + nodes[0], y[e] + nodes[i]);
      }
    }




  if ( mesh.RIGHTborder() ) 
    for ( int yy = 0; yy < local_dim; ++yy ) //RIGHT
    {

      int e = yy*local_dim + local_dim-2 ;

      for (int i=0; i<= N; ++i)
      {
        rhs(N,i,e)  += pen*weights[i]*u_ex(x[e] + nodes[N], y[e] + nodes[i]);
        for (int j=0; j<= N; ++j)
          rhs(j,i,e) -=    weights[i]*b.d_phi(j,N)*u_ex(x[e] + nodes[N], y[e] + nodes[i]);
      } 
    }

  

}


