#ifndef __CONJUGATE_GRADIENT_GPU_HPP__
#define __CONJUGATE_GRADIENT_GPU_HPP__

#include<abs_mvm.hpp>
#include<cublas_wrapper.hpp>
#include<conjugate_gradient_kernels.hpp>

#ifdef ONE_ITERATION_TEST
  #include<CUDA_TIMER.hpp>
#endif



/**
  This is the Conjugate Gradient function solver.   
*/
template<typename FLOAT_TYPE>
int conjugate_gradient ( const abs_mvm<FLOAT_TYPE> & problem, 
                         mode_vector<FLOAT_TYPE,int> x,
                         mode_vector<FLOAT_TYPE,int> b,
                         FLOAT_TYPE tol=1e-3)
{

  const int noe = x.get_noe();
  const int nompe = x.get_nompe();

  /* allocating space on device */
  mode_vector<FLOAT_TYPE, int> r(noe, nompe), p(noe, nompe), Ap(noe, nompe);

  const int blockD = 128;
  const int gridSIZE = (b.size() + blockD - 1)/blockD;

  cublasHandle_t handle;
  cublasCreate(&handle); 




  FLOAT_TYPE rr_old = .0, rr_new = .0 ;
  FLOAT_TYPE pAp = .0;

  // compute the L2-norm of b (to stop iterations)
  FLOAT_TYPE norm_b;
  cublas_dot(handle, b.size(), b.data(), 1, b.data(), 1, &norm_b);
  norm_b = sqrt(norm_b);

  // r = b - Ax
  problem._mvm(x, r);
  beta_kernel<<<gridSIZE, blockD>>>( b.size(), r.data(), b.data(), FLOAT_TYPE(-1) );

  /* p = r */
  cudaMemcpy (p.data(), r.data(), r.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);

  /* compute r*r */
  cublas_dot(handle, r.size(), r.data(), 1, r.data(), 1, &rr_old);





  const FLOAT_TYPE stop = /*norm_b*/tol;
     
  const int max_it = 10e8; // max iterations
  int it = 0; // iteration count


  while ( sqrt(rr_old)>stop && it < max_it )
  {

    // Ap = A*p
    problem._mvm(p, Ap);

    cublas_dot(handle, p.size(), p.data(), 1, Ap.data(), 1, &pAp);

    FLOAT_TYPE alpha = rr_old/pAp;

    // x(k+1) = x(k) + alpha*p(k)  
    // r(k+1) = r(k) - alpha*Ap(k)   
    alpha_kernel<<<gridSIZE, blockD>>>( x.size(), x.data(), p.data(), r.data(), Ap.data(), alpha );

    cublas_dot(handle, r.size(), r.data(), 1, r.data(), 1, &rr_new);

    FLOAT_TYPE beta = rr_new/rr_old;

    // p(k+1) = r(k+1) + beta*p(k)     
    beta_kernel<<<gridSIZE, blockD>>>( p.size(), p.data(), r.data(), beta );

    rr_old = rr_new;
    ++it;

  }   

  /* free space */
  cublasDestroy(handle);

  r.free();
  p.free();
  Ap.free();

  return it;
}






template<typename FLOAT_TYPE>
int preconditioned_conjugate_gradient ( const abs_mvm<FLOAT_TYPE> & problem, 
                                        mode_vector<FLOAT_TYPE,int> x,
                                        mode_vector<FLOAT_TYPE,int> b,
                                        FLOAT_TYPE tol=1e-3)
{

  const FLOAT_TYPE one(1), minus_one(-1); //, zero(0);

  const int noe = x.get_noe();
  const int nompe = x.get_nompe();


  /* allocating space on device */
  mode_vector<FLOAT_TYPE, int> r(noe, nompe), p(noe, nompe), Ap(noe, nompe), z(noe, nompe);

  const int blockD = 128;
  const int gridSIZE = (b.size() + blockD - 1)/blockD;

  cublasHandle_t handle;
  cublasCreate(&handle); 



  FLOAT_TYPE rr_old = .0, rr_new = .0 ;
  FLOAT_TYPE pAp = .0;


  // compute the L2-norm of b (to stop iterations)
  FLOAT_TYPE norm_b;
  cublas_dot(handle, b.size(), b.data(), 1, b.data(), 1, &norm_b);
  norm_b = sqrt(norm_b);

  // r = b - Ax
  problem._mvm(x, r);
  beta_kernel<<<gridSIZE, blockD>>>( b.size(), r.data(), b.data(), FLOAT_TYPE(-1) );

  problem._prec_mvm(r, z);
  // p = z 
  cudaMemcpy (p.data(), z.data(), z.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);

  // compute r*z 
  cublas_dot(handle, r.size(), r.data(), 1, z.data(), 1, &rr_old);


  const FLOAT_TYPE stop = /*norm_b*/tol;
  const int max_it = 10e8; // max iterations
  int it = 0; // iteration count
  FLOAT_TYPE rr(1);

  while ( sqrt(rr)>stop && it < max_it )
  {

    // Ap = A*p
    problem._mvm(p, Ap);

    cublas_dot(handle, p.size(), p.data(), 1, Ap.data(), 1, &pAp);

    FLOAT_TYPE alpha = rr_old/pAp;

    // x(k+1) = x(k) + alpha*p(k)  
    // r(k+1) = r(k) - alpha*Ap(k)   
    alpha_kernel<<<gridSIZE, blockD>>>( x.size(), x.data(), p.data(), r.data(), Ap.data(), alpha );

    cublas_dot(handle, r.size(), r.data(), 1, r.data(), 1, &rr);

    problem._prec_mvm(r, z);

    cublas_dot(handle, r.size(), z.data(), 1, r.data(), 1, &rr_new);

    FLOAT_TYPE beta = rr_new/rr_old;

    // p(k+1) = z(k+1) + beta*p(k)     
    beta_kernel<<<gridSIZE, blockD>>>( p.size(), p.data(), z.data(), beta );

    rr_old = rr_new;
    ++it;

  }   

  /* free space */
  cublasDestroy(handle);

  z.free();
  r.free();
  p.free();
  Ap.free();

  return it;
}



#endif

