/***
* /file 
*
* Validation test for 1d Laplacian matrix.
* 
*   solve  - u" = f  with Dirichlet border conditions
*   The numerical solution is confronted with the analytical one.
*
*/

#include<iostream>
#include<algorithm>
#include<cmath>

#include<lapack_wrapper.hpp>

#include<laplacian_operator.hpp>
#include<sem_function.hpp>


// ************************** function prototypes  ************************** 

template<typename FLOAT_TYPE>
void laplacian_matrix_1d (int N, FLOAT_TYPE * a);

template<typename FLOAT_TYPE>
int solve_poisson_1d_dirichlet_bc (int order,
                                   std::vector<FLOAT_TYPE> & solution,
                                   FLOAT_TYPE & H1_err,
                                   FLOAT_TYPE & L2_err,
                                   FLOAT_TYPE & max_err,
                                   int & index_max_err); 



// ****************************** problem data ****************************** 

/** exact solution */
template<typename FLOAT_TYPE>
FLOAT_TYPE u_ex (FLOAT_TYPE x)
{
  const FLOAT_TYPE PI = 3.14159265359;
 // return ( 1 - x*x*x*x )*x/5.;

 // return std::sin(4.*PI*x);

//  return std::cos(4.*PI*x);


 const FLOAT_TYPE A = 2; // u_ex(-1)
 const FLOAT_TYPE B = 1; // u_ex(1)

 return .5*(A+B+1) + .5*(B-A)*x - .5*x*x;

}



template<typename FLOAT_TYPE>
FLOAT_TYPE grad_u_ex (FLOAT_TYPE x)
{

 const FLOAT_TYPE A = 2; // u_ex(-1)
 const FLOAT_TYPE B = 1; // u_ex(1)

 return .5*(B-A) - x;


}


/** right-hand term */
template<typename FLOAT_TYPE>
FLOAT_TYPE f (FLOAT_TYPE x)
{
  const FLOAT_TYPE PI = 3.14159265359;
 // return 4.*x*x*x;

//  return 16.*PI*PI*std::sin(4.*PI*x);
//  return 16.*PI*PI*std::cos(4.*PI*x);

  return 1.;
}





// ************************************************************************** 


int main()
{

  const int order = 8;

//  for (int order = 2; order < 64; ++order)
  {

    std::vector<double> u;
    double max_err;
    double L2_err, H1_err;
    int idx_err;

    int succ = solve_poisson_1d_dirichlet_bc 
                (order, u, H1_err, L2_err, max_err, idx_err);

    if (succ != 0)
    {
      std::cout<<" gevs returned: "<<succ<<std::endl;
      assert(0);
    }

    std::cout<<order<<"\t"<<max_err<<"\t"<<L2_err<<"\t"<<H1_err<<std::endl;

  }

  return 0;
}





// ******************************** functions ******************************* 



template<typename FLOAT_TYPE>
void laplacian_matrix_1d (int N, std::vector<FLOAT_TYPE> & A)
{
  A.resize((N+1)*(N+1));

  laplace_1d<FLOAT_TYPE> lap(N);
 
  for (int i=0; i <= N; ++i)
    for (int j=0; j <= N; ++j)
      A[i*(N+1) + j] = lap(i,j);

}


template<typename T>
void dirichlet_bc_nitsche_method (int N, std::vector<T> & A)
{

  sem_function<T> b(N);

  for (int i=0; i <= N; ++i)
  {

    A[i]           += b.d_phi(i,0);
    A[i*(N+1)]     += b.d_phi(i,0);

    A[i*(N+1) + N] -= b.d_phi(i,N);
    A[N*(N+1) + i] -= b.d_phi(i,N); 

  }

  const double pen = 1e10;
  A[0]          -= pen;
  A[A.size()-1] += pen;

}


template<typename T>
void bc_nitsche_method_rhs (int N, std::vector<T> & rhs)
{

  sem_function<T> b(N);


  for (int i=0; i <= N; ++i)
    rhs[i] += b.d_phi(i,0)*u_ex(-1.) -  b.d_phi(i,N)*u_ex(1.) ;

  const double pen = 1e10;

  rhs[0] -= pen*u_ex(-1.); 
  rhs[N] += pen*u_ex(1.); 

}



template<typename T>
int solve_poisson_1d_dirichlet_bc (int order,
                                               std::vector<T> & solution,
                                               T & H1_err,
                                               T & L2_err,
                                               T & max_err,
                                               int & index_max_err )
{

  const int n = order;
  std::vector<int> ipiv(n+1);

  // ********************************** A **********************************

  std::vector<T> A;
  laplacian_matrix_1d<T>(n, A);

  dirichlet_bc_nitsche_method (order, A);

#if 0
  for (int i = 0; i <= n; ++i)
  {
    for (int j = 0; j <= n; ++j)
      std::cerr<<A[i*(n+1) + j ]<<"\t";

    std::cerr<<std::endl;
  }
#endif

  // ********************************** b **********************************

  std::vector<T> b(n+1);


  LGL_quadrature_table<T> qt(n+1);

  for (int i = 0; i < qt.nop(); ++i)
    b[i] = qt.weight(i)*f(qt.node(i));

  bc_nitsche_method_rhs(n, b); 

#if 0
  for (int i = 0; i <= n; ++i)
    std::cerr<<b[i]<<std::endl;
#endif


  // ****************************** solve Ax=b *****************************

  int info = lapacke_gesv( LAPACK_ROW_MAJOR, n+1, 1, A.data(), n+1, ipiv.data(), b.data(), 1 );

  // NOW b contains the solution
  solution = b;

#if 1
  for (int i = 0; i <=n; ++i)
    std::cerr<<qt.node(i)<<"\t"<<solution[i]<<"\t"<<u_ex(qt.node(i))<<std::endl;
#endif

  std::vector<T> err;

  for (int i = 0; i <= n; ++i)
    err.push_back (std::fabs(b[i] - u_ex(qt.node(i))));

  typename std::vector<T>::iterator m = std::max_element(err.begin() , err.end() );

  L2_err = 0;
  for (int i = 0; i <=n; ++i) L2_err += qt.weight(i)*err[i]*err[i]; 

  T L2_square_err = L2_err;

  L2_err = std::sqrt(L2_err);

  max_err = *m;

  index_max_err = m - err.begin(); 


  // ****** H^1 err norm *****


  std::vector<T> grad_err;

  sem_function<T> base(n);

  for (int i = 0; i <= n; ++i)
  {
   
    T eee(0);

    for (int p = 0; p <= n; ++p)
      eee += b[p]*base.d_phi(p,i) ;
 
    grad_err.push_back (std::fabs(eee - grad_u_ex(qt.node(i))));

  }

  H1_err = 0;
  for (int i = 0; i <=n; ++i) H1_err += qt.weight(i)*grad_err[i]*grad_err[i]; 

  H1_err += L2_square_err;

  H1_err = std::sqrt(H1_err); 

  
  return info;
}


