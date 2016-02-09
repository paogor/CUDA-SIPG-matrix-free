/***
* /file 
*
* Validation test for Laplacian matrix.
* 
*   solve  - u" = f  with Dirichlet boundary conditions
*   Numerical solution is confronted with analytical one.
*
*/

#include<iostream>
#include<algorithm>
#include<cmath>

#include<cublas_wrapper.hpp>

#include<mode_vector.hpp>
#include<LGL_quadrature_table.hpp>
#include<sem_function.hpp>
#include<sem_gpu_kernels.hpp>


#include<build_square_mesh.hpp>

template<typename FLOAT_TYPE>
void copy_and_show (mode_vector<FLOAT_TYPE, int> target)
{
  host_mode_vector<FLOAT_TYPE, int> h_target(target);

  for(int i1 = 0; i1 < h_target.get_nompe(); ++i1)
    for(int i2 = 0; i2 < h_target.get_nompe(); ++i2)
    {
      for(int e = 0; e < h_target.get_noe(); ++e)
        std::cerr<<h_target(i1,i2,e)<<"\t";
      std::cerr<<std::endl;
    }
}


template<typename T>
T u_ex (T x, T y)
{
  const T PI = 3.14159265359;
//  return x*y*(y-1)*(x-1)*std::exp(x-y);   // A
//  return x*(1-x)*std::exp(3*x)*std::sin(2*PI*y);  // B


//  return y*(1-y)*x*x*x; // C
//  return x*x*y*y; //D
  return std::cos(8*PI*x) + std::cos(8*PI*y); //E

}


template<typename T>
T f (T x, T y)
{
  const T PI = 3.14159265359;
//  return -2*x*(y-1)*(y-2*x+x*y+2)*std::exp(x-y); // A
//  return -std::sin(2*PI*y)*std::exp(3*x)*((4-3*x-9*x*x)-4*PI*PI*(x-x*x)); // B

//  return 2*x*x*x - 6*x*y*(1-y); // C 

//  return -2*y*y -2*x*x; //D
  return 64*PI*PI*std::cos(8*PI*x) + 64*PI*PI*std::cos(8*PI*y); //E
//  return 1.;
}

template<typename FLOAT_TYPE>
int conjugate_gradient (int order,
                        mesh_info<FLOAT_TYPE> mesh,
                        mode_vector<FLOAT_TYPE,int> x,
                        mode_vector<FLOAT_TYPE,int> b,
                        FLOAT_TYPE tol=1e-30)
{

  load_Dphi_table<double>(order);
  load_lgl_quadrature_table<double>(order);

  const FLOAT_TYPE one(1), minus_one(-1); //, zero(0);

  const int noe = x.get_noe();
  const int nompe = x.get_nompe();


  /* allocating space on device */
  mode_vector<FLOAT_TYPE, int> r(noe, nompe), p(noe, nompe), Ap(noe, nompe);

  cublasHandle_t handle;
  cublasCreate(&handle); 

  /* inizialize r: r=b-Ax */

  // r = -Ax
  mvm (order, mesh, x, r);
// copy_and_show (r);

  cublas_scal(handle, r.size(), &minus_one, r.data(), 1);
  //

  // r = b + r
  cublas_axpy(handle, b.size(), &one, b.data(), 1, r.data(), 1);
  /* p = r */
  cudaMemcpy (p.data(), r.data(), r.size()*sizeof(FLOAT_TYPE), cudaMemcpyDeviceToDevice);

//  copy_and_show (p);

  FLOAT_TYPE rr_old = .0, rr_new = .0 ;
  FLOAT_TYPE pAp = .0;

  /* compute r*r */
  cublas_dot(handle, r.size(), r.data(), 1, r.data(), 1, &rr_old);

//  std::cerr<<"rr_old "<<rr_old<<std::endl;


  const int max_it = 10e8; // max iterations
  int it = 0; // iteration count


//  while ( rr_old>tol && it < max_it )
  {

    /* compute Ap */

    // Ap = A*p
    mvm (order, mesh, p, Ap);
    //
//    copy_and_show (Ap);

    cublas_dot(handle, p.size(), p.data(), 1, Ap.data(), 1, &pAp);

    FLOAT_TYPE alpha = rr_old/pAp;
    FLOAT_TYPE minus_alpha = -1.*alpha;

    /* x(k+1) = x(k) + alpha*p(k) */ 
    cublas_axpy(handle, p.size(), &alpha, p.data(), 1, x.data(), 1);

//    copy_and_show (x);

    /* r */
    cublas_axpy(handle, Ap.size(), &minus_alpha, Ap.data(), 1, r.data(), 1);
    cublas_dot(handle, r.size(), r.data(), 1, r.data(), 1, &rr_new);

    FLOAT_TYPE beta = rr_new/rr_old;
    
    cublas_scal(handle, p.size(), &beta, p.data(), 1);
    cublas_axpy(handle, r.size(), &one, r.data(), 1, p.data(), 1);

//    std::cerr<<it<<": rr_old "<<rr_old<<std::endl;

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




#include<rhs.hpp>
#include<iomanip>

template<typename T>
int poisson_2d_dirichlet_bc( int order,
                             const square_mesh<T> & mesh,
                             T & L2_err,
                             T & max_err,
                             T & index_max_err )
{

  const int dim = mesh.dim();
  std::vector<int> neig = mesh.neighbourhood();
  std::vector<T>  x = mesh.x_coord();
  std::vector<T>  y = mesh.y_coord();
 
  const int noe = mesh.noe();

  // ********************************** b **********************************

  host_mode_vector<T,int> d_bb(noe, order+1);

  std::vector<T> nodes(order+1), weights(order+1);
  LGL_quadrature_table<T> qt(order+1);

  for (int i = 0; i <= order; ++i)
  {
    nodes[i] = (qt.node(i)*.5 + .5)/dim;
    weights[i] = qt.weight(i)*.5/dim;
  }

// ---------------------
  bc_nitsche_method_rhs (order, mesh, d_bb);

// ---------------------

  // ****************************** solve Ax=b *****************************

  host_mode_vector<T,int> d_xx(noe, order+1);

  mode_vector<T,int> xx(d_xx);
  mode_vector<T,int> bb(d_bb);
  

  // run function to solve
  int it = conjugate_gradient(order, mesh.device_info, xx, bb );

  // copy back the solution 
  copy(xx, d_xx);

  std::vector<T> err;

  for (int e = 0; e < noe; ++e)
    for (int i = 0; i <= order; ++i)
      for (int j = 0; j <= order; ++j)
        err.push_back ( std::fabs(d_xx(i,j,e) - u_ex(x[e] + nodes[i], y[e] + nodes[j])));

  typename std::vector<T>::iterator m = std::max_element(err.begin() , err.end() );

  L2_err = 0;
  for (int e = 0; e < noe; ++e)
    for (int i = 0; i <= order; ++i)
      for (int j = 0; j <= order; ++j)
        L2_err +=  weights[i]*weights[j]*std::pow(d_xx(i,j,e) - u_ex(x[e] + nodes[i], y[e] + nodes[j]), 2);


  L2_err = std::sqrt(L2_err);

  max_err = *m;

  index_max_err = m - err.begin(); 

#if 0
  for (int e = 0; e < noe; ++e)
  {
    for (int i = 0; i <= order; ++i)
      for (int j = 0; j <= order; ++j)
        std::cerr<<std::setw(10)<<x[e] + nodes[i]<<"\t"<<std::setw(10)<<y[e] + nodes[j]<<"\t"
                 <<std::setw(10)<<d_xx(i,j,e)<<"\t"<<std::setw(10)
                 <<u_ex(x[e] +nodes[i],y[e] + nodes[j])<<std::endl;

    std::cerr<<std::endl;
  }
#endif

  xx.free();
  bb.free();

  return it;
}




void test(int degree, int dim)
{

  square_mesh<double> sq_mesh(dim);

  double L2_err, max_err, index_max_err; 

  int it = poisson_2d_dirichlet_bc( degree, sq_mesh,
                                      L2_err, max_err, index_max_err );

  std::cout<<dim<<"\t"<<degree<<"\t";
  std::cout<<L2_err<<"\t"<<max_err<<"\t"<<it<<std::endl;

  sq_mesh.device_info.free();

}






int main()
{

#if 0

  test(8,8);
  test(4,16);
  test(2,32);

#else

  test(16,64);
  test(8,128);
  test(4,256);
  test(2,512);

#endif

  return 0;

}


