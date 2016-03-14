#ifndef __LAPACK_WRAPPER_CPP__
#define __LAPACK_WRAPPER_CPP__

#include<lapacke.h>

lapack_int lapacke_gesv ( int matrix_order,
                          lapack_int n,
                          lapack_int nrhs,
                          float* a, lapack_int lda,
                          lapack_int* ipiv,
                          float* b, lapack_int ldb )
{
  return LAPACKE_sgesv( matrix_order, n, nrhs, a, lda, ipiv, b, ldb );
}


lapack_int lapacke_gesv ( int matrix_order,
                          lapack_int n,
                          lapack_int nrhs,
                          double* a, lapack_int lda,
                          lapack_int* ipiv,
                          double* b, lapack_int ldb ) 

{
  return LAPACKE_dgesv( matrix_order, n, nrhs, a, lda, ipiv, b, ldb );
}


lapack_int lapacke_inv(double* A, int size)
{
  int *ipiv = new int[size+1];
  lapack_int info;

  info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, size, size, A, size, ipiv);
  if (info!=0) return info;

  LAPACKE_dgetri(LAPACK_ROW_MAJOR, size, A, size, ipiv);

  delete ipiv;
  return info;
}

#endif
  
