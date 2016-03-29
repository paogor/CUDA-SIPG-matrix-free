#ifndef __CONJUGATE_GRADIENT_MULTIGPU_HPP__
#define __CONJUGATE_GRADIENT_MULTIGPU_HPP__

#include<abs_mvm_multigpu.hpp>
#include<cublas_wrapper.hpp>
#include<check_nan.hpp>

template<typename FLOAT_TYPE>
int conjugate_gradient_multigpu ( abs_mvm_multigpu<FLOAT_TYPE> & problem, 
                         mode_vector<FLOAT_TYPE,int> x,
                         mode_vector<FLOAT_TYPE,int> b,
                         FLOAT_TYPE tol)
{

  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

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

  const FLOAT_TYPE stop = tol*tol;
  const int max_it = b.size(); // max iterations
  int it = 0; // iteration count


  while ( (rr_old)> stop && it < max_it )
  {

    // Ap = A*p
    problem._mvm(p);

/*    if ( check_nan(problem._mvm_output()) > 0 )
    {
      std::cout<<pid<<"\t"<<it<<std::endl;
    }
*/

    cudaDeviceSynchronize();

    pAp = problem._dot_product(problem._mvm_output(), p);

//    if (pid == 0) std::cout<<it<<": pAp "<<pAp<<std::endl;

    FLOAT_TYPE alpha = rr_old/pAp;
    FLOAT_TYPE minus_alpha = -1.*alpha;

//    if (pid == 0) std::cout<<it<<": alpha "<<alpha<<std::endl;
    // x(k+1) = x(k) + alpha*p(k)  
    cublas_axpy(handle, p.size(), &alpha, p.data(), 1, x.data(), 1);
    // r = r - alpha*Ap
    cublas_axpy(handle, problem._mvm_output().size(), &minus_alpha, problem._mvm_output().data(), 1, r.data(), 1);

    rr_new = problem._dot_product(r,r);

//    if (pid == 0) std::cout<<it<<": rr_new "<<rr_new<<std::endl;

    FLOAT_TYPE beta = rr_new/rr_old;

//    if (pid == 0) std::cout<<it<<": beta "<<beta<<std::endl;
    // p(k+1) = r(k+1) + beta*p(k)     
    cublas_scal(handle, p.size(), &beta, p.data(), 1);
    cublas_axpy(handle, r.size(), &one, r.data(), 1, p.data(), 1);

    rr_old = rr_new;
    ++it;

//   if (pid == 0) std::cout<<it<<": "<<rr_old<<std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
  }   

  cublasDestroy(handle);

  r.free();
  p.free();
  Ap.free();

  return it;

}


template<typename FLOAT_TYPE>
int precoditioned_conjugate_gradient_multigpu ( abs_mvm_multigpu<FLOAT_TYPE> & problem, 
                                                mode_vector<FLOAT_TYPE,int> x,
                                                mode_vector<FLOAT_TYPE,int> b,
                                                FLOAT_TYPE tol )
{
  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  const FLOAT_TYPE one(1), minus_one(-1); 

  const int noe = x.get_noe();
  const int nompe = x.get_nompe();

  /* allocating space on device */
  mode_vector<FLOAT_TYPE, int> r(noe, nompe), p(noe, nompe), Ap(noe, nompe), z(noe, nompe);

  cublasHandle_t handle;
  cublasCreate(&handle); 

  // compute the L2-norm of b (to stop iterations)
  FLOAT_TYPE norm_b = problem._dot_product(b,b);
  norm_b = sqrt(norm_b);

  // r = b 
  cudaMemcpy (r.data(), b.data(), b.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);

  problem._prec_mvm(r, z);
  // p = z 
  cudaMemcpy (p.data(), z.data(), z.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);


  FLOAT_TYPE rr_old = .0, rr_new = .0 ;
  FLOAT_TYPE pAp = .0;

  /* compute r*r */
  rr_old = problem._dot_product(r,z);

//  if (pid == 0) std::cout<<rr_old<<std::endl;

  const FLOAT_TYPE stop = tol*tol;
  const int max_it = b.size(); // max iterations
  int it = 0; // iteration count

  FLOAT_TYPE rr(1);

  while ( (rr)> stop && it < max_it )
  {

    // Ap = A*p
    problem._mvm(p);
    pAp = problem._dot_product(problem._mvm_output(), p);

//    if (pid == 0) std::cout<<it<<": pAp "<<pAp<<std::endl;
    FLOAT_TYPE alpha = rr_old/pAp;
    FLOAT_TYPE minus_alpha = -1.*alpha;

//    if (pid == 0) std::cout<<it<<": alpha "<<alpha<<std::endl;
    // x(k+1) = x(k) + alpha*p(k)  
    cublas_axpy(handle, p.size(), &alpha, p.data(), 1, x.data(), 1);


    // r 
    cublas_axpy(handle, problem._mvm_output().size(), &minus_alpha, problem._mvm_output().data(), 1, r.data(), 1);
    rr = problem._dot_product(r,r);
//    if (pid == 0) std::cout<<it<<": rr "<<rr<<std::endl;

    // REDUCE sum rr_new

    problem._prec_mvm(r, z);
    rr_new = problem._dot_product(r,z);
//    if (pid == 0) std::cout<<it<<": rr_new "<<rr_new<<std::endl;

    FLOAT_TYPE beta = rr_new/rr_old;
//    if (pid == 0) std::cout<<it<<": beta "<<beta<<std::endl;

    // p(k+1) = z(k+1) + beta*p(k)     
    cublas_scal(handle, p.size(), &beta, p.data(), 1);
    cublas_axpy(handle, z.size(), &one, z.data(), 1, p.data(), 1);


    rr_old = rr_new;
    ++it;

    MPI_Barrier(MPI_COMM_WORLD);
//   if (pid == 0) std::cout<<it<<": "<<beta<<std::endl;

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

