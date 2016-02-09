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


#endif
 
