#ifndef __LAPLACIAN_OPERATOR_HPP__XIIHF
#define __LAPLACIAN_OPERATOR_HPP__XIIHF


#include<sem_function.hpp>

#ifdef USE_PRECONDITIONER
  #include<lapack_wrapper.hpp>
#endif

#define SEM_FUNC sem_function<FLOAT_TYPE>

/**
 The class defines the mono-dimensional Laplace operator.
 *This class is used for tests.*
*/
template<typename FLOAT_TYPE>
class laplace_1d : public SEM_FUNC
{

  public:

  laplace_1d(int degree) : SEM_FUNC(degree) {};

  /**

  /param i row
  /param j col
  /return (i,j) value of Laplace operator
  */
  FLOAT_TYPE operator() (int i, int j)
  {
   
    FLOAT_TYPE r = 0; 

    for (int z = 0; z < SEM_FUNC::_qt.nop(); ++z)
      r +=   SEM_FUNC::_qt.weight(z)
           * SEM_FUNC::d_phi(i,z)
           * SEM_FUNC::d_phi(j,z) ;
     
    return r;

  }

};  

/**
 The class defines the two-dimensional Laplace operator.
 *This class is used for tests.*
*/
template<typename FLOAT_TYPE>
class laplace_2d : public SEM_FUNC
{

  public:

  laplace_2d(int degree) : SEM_FUNC(degree) {};

  /**

  \param i1
  \param i2
  \param j1
  \param j2
  \return ([i1,i2],[j1,j2]) value of Laplace operator
  */
  FLOAT_TYPE operator() (int i1, int i2, int j1,  int j2)
  {
    
    FLOAT_TYPE result = 0;

    if (i2 == j2)
      for (int k = 0; k < SEM_FUNC::_qt.nop(); ++k)
        result +=  SEM_FUNC::_qt.weight(k)
                  *SEM_FUNC::_qt.weight(i2)
                  *SEM_FUNC::d_phi(i1, k)
                  *SEM_FUNC::d_phi(j1, k);

    if (i1 == j1)
      for (int k = 0; k < SEM_FUNC::_qt.nop(); ++k)
        result +=  SEM_FUNC::_qt.weight(k)
                  *SEM_FUNC::_qt.weight(i1)
                  *SEM_FUNC::d_phi(i2, k)
                  *SEM_FUNC::d_phi(j2, k);

    return result;

  }

};  




#ifdef USE_PRECONDITIONER
class block_diagonal_preconditioner_2d 
{

  private:

  int nn;
  int degree;
  void build_diagonal_block();
  double pen;

  int inDEX(int N, int i1, int i2, int j1, int j2)
  {
//    return (N+1)*(N+1)*((N+1)*j2 + j1) + ((N+1)*i2 + i1);
    return (N+1)*(N+1)*((N+1)*i2 + i1) + ((N+1)*j2 + j1);
  }

  public:

  std::vector<double> a;
  block_diagonal_preconditioner_2d(int _degree, int _pen)
   : degree(_degree),
     nn((_degree+1)*(_degree+1)),
     a((_degree+1)*(_degree+1)*(_degree+1)*(_degree+1), 0),
     pen(_pen)
  {
    build_diagonal_block();
    lapacke_inv(a.data(), nn);
  }

  double operator() (int i1, int i2, int j1,  int j2)
  {
    return a[inDEX(degree, i1, i2, j1, j2)]; 
  }


};


void block_diagonal_preconditioner_2d::build_diagonal_block()
{
  const int N = degree;

  sem_function<double> b(N);
  laplace_2d<double> lap(N);
  LGL_quadrature_table<double> qt(N+1);

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
      a[ inDEX(N,i,j,0,j) ] -= -1*qt.weight(j)*b.d_phi(i,0)*.5;
      a[ inDEX(N,0,j,i,j) ] -= -1*qt.weight(j)*b.d_phi(i,0)*.5;

      //RIGHT
      a[ inDEX(N,i,j,N,j) ] -= qt.weight(j)*b.d_phi(i,N)*.5;
      a[ inDEX(N,N,j,i,j) ] -= qt.weight(j)*b.d_phi(i,N)*.5;

      //DOWN
      a[ inDEX(N,j,i,j,0) ] -= -1*qt.weight(j)*b.d_phi(i,0)*.5;
      a[ inDEX(N,j,0,j,i) ] -= -1*qt.weight(j)*b.d_phi(i,0)*.5;

      //UP
      a[ inDEX(N,j,i,j,N) ] -= qt.weight(j)*b.d_phi(i,N)*.5;
      a[ inDEX(N,j,N,j,i) ] -= qt.weight(j)*b.d_phi(i,N)*.5;
    } 


  for (int i=0; i <= N; ++i)
  {
    a[ inDEX(N,N,i,N,i) ] += pen*qt.weight(i);
    a[ inDEX(N,i,N,i,N) ] += pen*qt.weight(i);

    a[ inDEX(N,0,i,0,i) ] += pen*qt.weight(i);
    a[ inDEX(N,i,0,i,0) ] += pen*qt.weight(i);
  }

}
#endif 


#endif 


