/***
* /file 
*
* Validation test for Laplacian matrix.
* 
*   solve  - u" = f  with Dirichlet border conditions
*   The numerical solution is confronted with the analytical one.
*
*
*
*/

#include<iostream>
#include<algorithm>
#include<cmath>

#include<lapack_wrapper.hpp>

#include<laplacian_operator.hpp>
#include<sem_function.hpp>


template<typename T> void laplacian_matrix_2d(int N, std::vector<T> & a);

template<typename T>
int poisson_2d_dirichlet_bc( int order,
                                         std::vector<T> & solution,
                                         T & H1_err,
                                         T & L2_err,
                                         T & max_err, 
                                         T & index_max_err );

#define EXACT_SOLUTION_NO 5
#include"../analytical_solutions.hpp"
using namespace test_func;


int main()
{

  std::cout<<"EXACT_SOLUTION_NO "<<EXACT_SOLUTION_NO<<std::endl;

  for (int order = 2; order < 40; ++order)
  {

    std::vector<double> u;
    double max_err, H1_err, L2_err, index_max_err;

    int succ = poisson_2d_dirichlet_bc
               (order, u, H1_err, L2_err, max_err, index_max_err);

    if (succ != 0)
    {
      std::cout<<" gevs returned: "<<succ<<std::endl;
      assert(0);
    }

    std::cout<<order<<"\t&\t"<<H1_err<<"\t&\t"<<L2_err<<"\t&\t"<<max_err<<" \\\\"<<std::endl;

  }

  return 0;
}


int inDEX(int N, int i1, int i2, int j1, int j2)
{
  return (N+1)*(N+1)*((N+1)*j2 + j1) + ((N+1)*i2 + i1) ;
}


int vec_INDEX(int N, int i1, int i2)
{
  return ((N+1)*i2 + i1) ;
}



template<typename T>
void laplacian_matrix_2d(int N, std::vector<T> & a)
{

  laplace_2d<T> lap(N);

  int append = 0;

  for (int i1=0; i1 <= N; ++i1)
    for (int i2=0; i2 <= N; ++i2)
      for (int j1=0; j1 <= N; ++j1)
        for (int j2=0; j2 <= N; ++j2)
          a[ inDEX(N,i1,i2,j1,j2) ] = lap(i1,i2,j1,j2);

}


template<typename T>
void dirichlet_bc_nitsche_method_2d (int N, std::vector<T> & a)
{

  sem_function<T> b(N);
  std::vector<T> nodes(N+1), weights(N+1);

  LGL_quadrature_table<T> qt(N+1);

  for (int i = 0; i <= N; ++i)
  {
    nodes[i] = qt.node(i)*.5 + .5;
    weights[i] = qt.weight(i)*.5;
  }

  // DOWN edge Dphi_i*phi_j
  // j (0,0)  i ([0..N],0)
  // j (..,0) i ([0..N],0)
  // j (N,0)  i ([0..N],0)

  for (int j=0; j <= N; ++j)
  for (int i=0; i <= N; ++i)
  {
    //LEFT
    a[ inDEX(N,i,j,0,j) ] -= -1*weights[j]*b.d_phi(i,0)*2;
    a[ inDEX(N,0,j,i,j) ] -= -1*weights[j]*b.d_phi(i,0)*2;

    //RIGHT
    a[ inDEX(N,i,j,N,j) ] -= weights[j]*b.d_phi(i,N)*2;
    a[ inDEX(N,N,j,i,j) ] -= weights[j]*b.d_phi(i,N)*2;

    //DOWN
    a[ inDEX(N,j,i,j,0) ] -= -1*weights[j]*b.d_phi(i,0)*2;
    a[ inDEX(N,j,0,j,i) ] -= -1*weights[j]*b.d_phi(i,0)*2;

    //UP
    a[ inDEX(N,j,i,j,N) ] -= weights[j]*b.d_phi(i,N)*2;
    a[ inDEX(N,j,N,j,i) ] -= weights[j]*b.d_phi(i,N)*2;
  } 


  const T pen = 10e1; // check
  for (int i=0; i <= N; ++i)
  {
    a[ inDEX(N,N,i,N,i) ] += pen*weights[i];
    a[ inDEX(N,i,N,i,N) ] += pen*weights[i];

    a[ inDEX(N,0,i,0,i) ] += pen*weights[i];
    a[ inDEX(N,i,0,i,0) ] += pen*weights[i];
  }


}


template<typename T>
void bc_nitsche_method_rhs (int N, std::vector<T> & rhs)
{

  sem_function<T> b(N);

  std::vector<T> nodes(N+1), weights(N+1);
  LGL_quadrature_table<T> qt(N+1);

  for (int i = 0; i <= N; ++i)
  {
    nodes[i] = qt.node(i)*.5 + .5;
    weights[i] = qt.weight(i)*.5;
  }

  for (int i=0; i<= N; ++i)
  for (int j=0; j<= N; ++j)
  {
    //DOWN
    rhs[ vec_INDEX(N,i,j) ] -= -1*weights[i]*b.d_phi(j,0)*u_ex(nodes[i],nodes[0])*2;

    //UP
    rhs[ vec_INDEX(N,i,j) ] -=    weights[i]*b.d_phi(j,N)*u_ex(nodes[i],nodes[N])*2;

    //LEFT
    rhs[ vec_INDEX(N,j,i) ] -= -1*weights[i]*b.d_phi(j,0)*u_ex(nodes[0],nodes[i])*2;

    //RIGHT
    rhs[ vec_INDEX(N,j,i) ] -=    weights[i]*b.d_phi(j,N)*u_ex(nodes[N],nodes[i])*2;
  } 

  const T pen = 10e1; // check
  for (int i=0; i <= N; ++i)
  {
    rhs[ vec_INDEX(N,N,i) ] += pen*weights[i]*u_ex(nodes[N],nodes[i]);
    rhs[ vec_INDEX(N,i,N) ] += pen*weights[i]*u_ex(nodes[i],nodes[N]);

    rhs[ vec_INDEX(N,0,i) ] += pen*weights[i]*u_ex(nodes[0],nodes[i]);
    rhs[ vec_INDEX(N,i,0) ] += pen*weights[i]*u_ex(nodes[i],nodes[0]);
  }


}



#include<iomanip>

template<typename T>
int poisson_2d_dirichlet_bc( int order,
                                         std::vector<T> & solution,
                                         T & H1_err,
                                         T & L2_err,
                                         T & max_err,
                                         T & index_max_err )
{

  const int vector_size = (order+1)*(order+1);
  const int matrix_size = vector_size*vector_size;

  std::vector<lapack_int> ipiv(vector_size, 0); // pivot

  // ********************************** A **********************************

  std::vector<T> A(matrix_size, 0); //< stiffness matrix
  laplacian_matrix_2d<T>(order, A);

  // penalty term for Diriclet boundary conditions
  // and scale operator for (0,1)x(0,1)

  dirichlet_bc_nitsche_method_2d (order, A);

  // ********************************** b **********************************

  std::vector<T> b(vector_size, 0); //< rhs vector

  std::vector<T> nodes(order+1), weights(order+1);
  LGL_quadrature_table<T> qt(order+1);

  // nodes scaling
  for (int i = 0; i <qt.nop(); ++i)
  {
    nodes[i] = qt.node(i)*.5 + .5;
    weights[i] = qt.weight(i)*.5;
  }

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      b[vec_INDEX(order,i,j)] = weights[i]*weights[j]*f(nodes[i], nodes[j]);

  bc_nitsche_method_rhs (order, b);

  // ****************************** solve Ax=b *****************************

  int info = lapacke_gesv( LAPACK_ROW_MAJOR, vector_size, 1, A.data(), vector_size, ipiv.data(), b.data(), 1 );

  // NOW b contains the solution
  solution = b;



  std::vector<T> err;

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      err.push_back ( std::fabs(b[vec_INDEX(order,i,j)] - u_ex(nodes[i], nodes[j])));

  typename std::vector<T>::iterator m = std::max_element(err.begin() , err.end() );

  L2_err = 0;
  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      L2_err +=  weights[i]*weights[j]*std::pow(b[vec_INDEX(order,i,j)] - u_ex(nodes[i], nodes[j]), 2);


  T L2_square_err = L2_err;


  L2_err = std::sqrt(L2_err);

  max_err = *m;

  index_max_err = m - err.begin(); 


#if 0

  for (int i = 0; i <= order; ++i)
    for (int j = 0; j <= order; ++j)
      std::cerr<<std::setw(10)<<nodes[i]<<"\t"<<std::setw(10)<<nodes[j]<<"\t"
               <<std::setw(10)<<b[vec_INDEX(order,i,j)]<<"\t"<<std::setw(10)
               <<u_ex(nodes[i],nodes[j])<<std::endl;
  
#endif


  sem_function<T> sf(order);

  H1_err = 0;

  T dx_square_err(0), dy_square_err(0);


  for (int p = 0; p <= order; ++p)     // node index on x-axis 
    for (int q = 0; q <= order; ++q)   // node index on y-axis 
    {

      T dx_u(0), dy_u(0);
      T weight2d = weights[p]*weights[q];

      for (int i = 0; i <= order; ++i)   // mode index on x-axis
        for (int j = 0; j <= order; ++j) // mode index on y-axis
        {

          dx_u += b[vec_INDEX(order,i,j)]*2*sf.d_phi(i,p)*sf.phi(j,q); 
          dy_u += b[vec_INDEX(order,i,j)]*2*sf.phi(i,p)*sf.d_phi(j,q); 

        }

      dx_square_err += weight2d*std::pow( dx_u - dx_u_ex(nodes[p], nodes[q]) ,2);
      dy_square_err += weight2d*std::pow( dy_u - dy_u_ex(nodes[p], nodes[q]) ,2); 

    }

  H1_err = L2_square_err + dx_square_err + dy_square_err;

  H1_err = std::sqrt(H1_err);

  return info;

}

