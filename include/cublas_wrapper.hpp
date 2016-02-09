/**
  \brief A simple CUBLAS wrapper.

  I overload the CUBLAS functions for handle types (double or float). 

*/

#ifndef __CUBLAS_WRAPPER_HPP__
#define __CUBLAS_WRAPPER_HPP__

#include<cublas_v2.h> 


cublasStatus_t cublas_scal (cublasHandle_t handle, 
                            int n, 
                            const double *alpha,
                            double *x, 
                            int incx)
{
  return cublasDscal (handle, n, alpha, x, incx);
}


cublasStatus_t cublas_scal (cublasHandle_t handle, 
                            int n, 
                            const float *alpha,
                            float *x, 
                            int incx)
{
  return cublasSscal (handle, n, alpha, x, incx);
}





cublasStatus_t cublas_axpy (cublasHandle_t handle,
                            int n, 
                            const double *alpha,
                            const double *x, 
                            int incx,
                            double *y,
                            int incy)
{
  return cublasDaxpy (handle, n, alpha, x, incx, y, incy);
}


cublasStatus_t cublas_axpy (cublasHandle_t handle,
                            int n, 
                            const float *alpha,
                            const float *x, 
                            int incx,
                            float *y,
                            int incy)
{
  return cublasSaxpy (handle, n, alpha, x, incx, y, incy);
}



cublasStatus_t cublas_dot (cublasHandle_t handle,
                           int n, 
                           const double *x, 
                           int incx, 
                           const double *y,
                           int incy,
                           double *result)
{
  return cublasDdot (handle, n, x, incx, y, incy, result);

} 

cublasStatus_t cublas_dot (cublasHandle_t handle,
                           int n, 
                           const float *x, 
                           int incx, 
                           const float *y,
                           int incy,
                           float *result)
{
  return cublasSdot (handle, n, x, incx, y, incy, result);

} 


cublasStatus_t cublas_gemv (cublasHandle_t handle, 
                            cublasOperation_t trans, 
                            int m,
                            int n,
                            const double *alpha, 
                            const double *A,
                            int lda,
                            const double *x,
                            int incx,
                            const double *beta,
                            double *y, 
                            int incy)
{
  return cublasDgemv (handle, trans, m, n, alpha, A, lda,
                                           x, incx, beta, y, incy);
}


cublasStatus_t cublas_gemv (cublasHandle_t handle, 
                            cublasOperation_t trans, 
                            int m,
                            int n,
                            const float *alpha, 
                            const float *A,
                            int lda,
                            const float *x,
                            int incx,
                            const float *beta,
                            float *y, 
                            int incy)
{
  return cublasSgemv (handle, trans, m, n, alpha, A, lda,
                                           x, incx, beta, y, incy);
}



#endif 
