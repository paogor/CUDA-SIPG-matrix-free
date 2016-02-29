#ifndef __CONJUGATE_GRADIENT_MULTIGPU_HPP__
#define __CONJUGATE_GRADIENT_MULTIGPU_HPP__

#include<abs_mvm_multigpu.hpp>
#include<cublas_wrapper.hpp>


/*

 x, the initial guess, is zero ?

 x doesn't need halos
 b doesn't need halos
 p needs halos

*/

/*
  exchange of the data between data need to be implement here
  because gcl needs to know a priori the pointer to insert it
  in a field
*/

template<typename FLOAT_TYPE>
int conjugate_gradient_multigpu ( abs_mvm_multigpu<FLOAT_TYPE> & problem, 
                         mode_vector<FLOAT_TYPE,int> x,
                         mode_vector<FLOAT_TYPE,int> b,
                         FLOAT_TYPE tol=1e-14)
{
  const FLOAT_TYPE one(1), minus_one(-1); //, zero(0);

  const int noe = x.get_noe();
  const int nompe = x.get_nompe();

  /* allocating space on device */
  mode_vector<FLOAT_TYPE, int> r(noe, nompe), p(noe, nompe), Ap(noe, nompe);

  cublasHandle_t handle;
  cublasCreate(&handle); 

  // compute the L2-norm of b (to stop iterations)
  FLOAT_TYPE norm_b = problem._dot_product(b,b);
  norm_b = sqrt(norm_b);

  // r = b 
  cudaMemcpy (r.data(), b.data(), b.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);
  // p = r 
  cudaMemcpy (p.data(), r.data(), r.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);


  FLOAT_TYPE rr_old = .0, rr_new = .0 ;
  FLOAT_TYPE pAp = .0;

  /* compute r*r */
  rr_old = problem._dot_product(r,r);

//  std::cerr<<"rr_old "<<rr_old<<std::endl;

  const FLOAT_TYPE stop = norm_b*tol * norm_b*tol;
  const int max_it = 10e8; // max iterations
  int it = 0; // iteration count


  while ( /*sqrt*/(rr_old)> stop && it < max_it )
  {

    /* compute Ap */

    // SYNC p

    // Ap = A*p
    problem._mvm(p);
    //
   
    pAp = problem._dot_product(problem._mvm_output(), p);

    // REDUCE sum pAp

    FLOAT_TYPE alpha = rr_old/pAp;
    FLOAT_TYPE minus_alpha = -1.*alpha;

    // x(k+1) = x(k) + alpha*p(k)  
    cublas_axpy(handle, p.size(), &alpha, p.data(), 1, x.data(), 1);


    // r 
    cublas_axpy(handle, problem._mvm_output().size(), &minus_alpha, problem._mvm_output().data(), 1, r.data(), 1);
    rr_new = problem._dot_product(r,r);

    // REDUCE sum rr_new

    FLOAT_TYPE beta = rr_new/rr_old;

    // p(k+1) = r(k+1) + beta*p(k)     
    cublas_scal(handle, p.size(), &beta, p.data(), 1);
    cublas_axpy(handle, r.size(), &one, r.data(), 1, p.data(), 1);

//    std::cerr<<it<<": rr_old "<<rr_old<<std::endl;

    rr_old = rr_new;
    ++it;


  }   

//  std::cerr<<it<<": rr_old "<<rr_old<<std::endl;

  /* free space */
  cublasDestroy(handle);

  r.free();
  p.free();
  Ap.free();

  return it;

}




#endif

