/***
* /file 
*
* Preconditioned attempt. 
*
*
*/

#define USE_PRECONDITIONER

#include<iostream>
#include<algorithm>
#include<cmath>

#include<lapack_wrapper.hpp>

#include<laplacian_operator.hpp>
#include<sem_function.hpp>



int inDEX(int N, int i1, int i2, int j1, int j2)
{
  return (N+1)*(N+1)*((N+1)*j2 + j1) + ((N+1)*i2 + i1) ;
}

template<typename T>
void diagonal_block (int N, std::vector<T> & a)
{

  sem_function<T> b(N);
  laplace_2d<T> lap(N);
  LGL_quadrature_table<T> qt(N+1);


  for (int i1=0; i1 <= N; ++i1)
    for (int i2=0; i2 <= N; ++i2)
      for (int j1=0; j1 <= N; ++j1)
        for (int j2=0; j2 <= N; ++j2)
          a[ inDEX(N,i1,i2,j1,j2) ] = lap(i1,i2,j1,j2);


  // DOWN edge Dphi_i*phi_j
  // j (0,0)  i ([0..N],0)
  // j (..,0) i ([0..N],0)
  // j (N,0)  i ([0..N],0)

  for (int j=0; j <= N; ++j)
    for (int i=0; i <= N; ++i)
    {
      //LEFT
      a[ inDEX(N,i,j,0,j) ] -= -1*qt.weight(j)*b.d_phi(i,0);
      a[ inDEX(N,0,j,i,j) ] -= -1*qt.weight(j)*b.d_phi(i,0);

      //RIGHT
      a[ inDEX(N,i,j,N,j) ] -= qt.weight(j)*b.d_phi(i,N);
      a[ inDEX(N,N,j,i,j) ] -= qt.weight(j)*b.d_phi(i,N);

      //DOWN
      a[ inDEX(N,j,i,j,0) ] -= -1*qt.weight(j)*b.d_phi(i,0);
      a[ inDEX(N,j,0,j,i) ] -= -1*qt.weight(j)*b.d_phi(i,0);

      //UP
      a[ inDEX(N,j,i,j,N) ] -= qt.weight(j)*b.d_phi(i,N);
      a[ inDEX(N,j,N,j,i) ] -= qt.weight(j)*b.d_phi(i,N);
    } 


  const T pen = 100*N*N;
  for (int i=0; i <= N; ++i)
  {
    a[ inDEX(N,N,i,N,i) ] += pen*qt.weight(i);
    a[ inDEX(N,i,N,i,N) ] += pen*qt.weight(i);

    a[ inDEX(N,0,i,0,i) ] += pen*qt.weight(i);
    a[ inDEX(N,i,0,i,0) ] += pen*qt.weight(i);
  }


}


template<typename T>
int diagonal_matrix_inversion( int order )
{

  const int nn = (order+1)*(order+1);

  block_diagonal_preconditioner_2d prec(order); 
  std::vector<T> A(nn*nn, 0); 
  diagonal_block<T>(order, A);
 
  std::vector<T> A1 = A;
  lapacke_inv(A1.data(), nn);


/*
  for (int i=0; i < nn; i++)
  {
    for (int j=0; j < nn; j++)
      std::cout<<"\t"<<A[i*nn+j];

    std::cout<<std::endl;
  }

    std::cout<<std::endl;
*/
  for (int i=0; i < nn; i++)
  {
    for (int j=0; j < nn; j++)
      std::cout<<"\t"<<A1[i*nn+j] - prec.a[i*nn+j];

    std::cout<<std::endl;
  }

/*
    std::cout<<std::endl;

  for (int i=0; i < nn; i++)
  {

    for (int j=0; j < nn; j++)
    {
      T r = 0;
      for (int k=0; k < nn; k++)
        r += A[i*nn+k]*A1[k*nn+j];
      std::cout<<"\t"<<r;
    }

    std::cout<<std::endl;

  }
*/
  return 0;

}






int main()
{


  diagonal_matrix_inversion<double>( 5 );

  return 0;
}


