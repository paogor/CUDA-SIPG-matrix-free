#ifndef __SIPG_SEM_2D_HPP__
#define __SIPG_SEM_2D_HPP__


#include<sem_function.hpp>
#include<conjugate_gradient_gpu.hpp>
#include<mode_vector.hpp>
#include<LGL_quadrature_table.hpp>
#include<sipg_sem_2d_gpu_kernels.hpp>
#include<build_square_mesh.hpp>

#include<../performance_tests/flux_kernels_optim.hpp>

#ifdef USE_MODE_MATRIX
  #include<mode_matrix_kernels.hpp>
#endif

#ifdef USE_PRECONDITIONER
  #include<mode_matrix_kernels.hpp>
#endif

#include<CUDA_TIMER.hpp>

/**
  This class solves the Poisson problem with Dirichlet border condition
  using the Symmetric Interior Penalty Galerkin method with Spectral Element
  Method basis.

  All the work is done by the constructor. It generates the right hand side,
  initializes the device memory, calls the Conjugate Gradient solver.
*/
template<typename FLOAT_TYPE>
class sipg_sem_2d : public abs_mvm<FLOAT_TYPE>
{

  private:

    int order;
    square_mesh<FLOAT_TYPE> mesh;
    LGL_quadrature_table<FLOAT_TYPE> qt;
    sem_function<FLOAT_TYPE> basis;

    FLOAT_TYPE pen; 

#ifdef USE_MODE_MATRIX
    mode_matrix<FLOAT_TYPE, int> d_volume_matrix; 
#endif

#ifdef USE_PRECONDITIONER
    mode_matrix<FLOAT_TYPE, int> d_prec_matrix; 
#endif

    
    CUDA_TIMER system_solution_time;

    void compute_rhs( FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
                      FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE) );

    void err_norms( host_mode_vector<FLOAT_TYPE,int> & u, 
                    FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
                    FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE),
                    FLOAT_TYPE (*dx_u_ex)(FLOAT_TYPE, FLOAT_TYPE),
                    FLOAT_TYPE (*dy_u_ex)(FLOAT_TYPE, FLOAT_TYPE) );

    dim3 volume_gridSIZE;
    dim3 volume_blockSIZE;

    dim3 flux_gridSIZE;
    dim3 flux_blockSIZE;

  public:

    FLOAT_TYPE max_err;
    FLOAT_TYPE L2_err;
    FLOAT_TYPE H1_err;
    int iterations; 

    mode_vector<FLOAT_TYPE,int> d_u;
    mode_vector<FLOAT_TYPE,int> d_rhs;

    host_mode_vector<FLOAT_TYPE,int> solution;

    sipg_sem_2d ( int _order, 
                  const square_mesh<FLOAT_TYPE> & _mesh,
                  FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
                  FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE), 
                  FLOAT_TYPE (*dx_u_ex)(FLOAT_TYPE, FLOAT_TYPE), 
                  FLOAT_TYPE (*dy_u_ex)(FLOAT_TYPE, FLOAT_TYPE),
                  FLOAT_TYPE _pen,
                  FLOAT_TYPE _tol )
     : qt(_order+1),
       basis(_order),
       pen(_pen)
    {

      const int noe = _mesh.dim()*_mesh.dim();
      order = _order;
      mesh = _mesh;
      // initialize
      load_Dphi_table<FLOAT_TYPE>(order);
      load_lgl_quadrature_table<FLOAT_TYPE>(order);

      const int blockD = 128;
      volume_gridSIZE = dim3( (noe + blockD - 1)/blockD , order+1, order+1);
      volume_blockSIZE = dim3(blockD, 1, 1); 

      const int dimx = mesh.device_info.get_dimx();
      const int dimy = mesh.device_info.get_dimy();
      const int blockDx = 32;
      const int blockDy = 4;

      flux_gridSIZE = dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 );
      flux_blockSIZE = dim3( blockDx, blockDy, 1 );


#ifdef USE_MODE_MATRIX
      host_laplacian_matrix<FLOAT_TYPE,int> h_volume_matrix(1, order);
      d_volume_matrix = h_volume_matrix;
#endif

#ifdef USE_PRECONDITIONER
      host_preconditioner_matrix<FLOAT_TYPE,int> h_prec_matrix(1, order, pen);
      d_prec_matrix = h_prec_matrix;
#endif

      host_mode_vector<FLOAT_TYPE,int> h_xx(noe, order+1);
      copy(h_xx, d_u);

      compute_rhs(f, u_ex);


      system_solution_time.start();
#ifdef USE_PRECONDITIONER
      iterations = preconditioned_conjugate_gradient(*(this), d_u, d_rhs, _tol);
#else
      iterations = conjugate_gradient(*(this), d_u, d_rhs, _tol);
#endif
      system_solution_time.stop();

      // copy back the solution 
      copy(d_u, solution);

      err_norms(solution, f, u_ex, dx_u_ex, dy_u_ex);

    }

    ~sipg_sem_2d ()
    {
#ifdef USE_MODE_MATRIX
      d_volume_matrix.free();
#endif
#ifdef USE_PRECONDITIONER
      d_prec_matrix.free();
#endif
      d_rhs.free();
      d_u.free();
    }

    float solution_time()
    {
      return system_solution_time.elapsed_millisecs();
    }


    void print_result();
           
    int _mvm ( mode_vector<FLOAT_TYPE,int> input,
               mode_vector<FLOAT_TYPE,int> output ) const
    {


#ifdef USE_MODE_MATRIX
      volume_mvm <FLOAT_TYPE, 1>
      <<<volume_gridSIZE, volume_blockSIZE>>>
      ( order, d_volume_matrix, input, output ); 
#else
      volume<FLOAT_TYPE>
      <<<volume_gridSIZE, volume_blockSIZE>>>
      ( order, input, output ); 
#endif

      flux_term6a<FLOAT_TYPE>
      <<<flux_gridSIZE, flux_blockSIZE>>>
      ( order, mesh.device_info, input, output, pen );

      flux_term6b<FLOAT_TYPE>
      <<<flux_gridSIZE, flux_blockSIZE>>>
      ( order, mesh.device_info, input, output, pen );


    #if 0

      cudaError_t error = cudaGetLastError();
      std::string lastError = cudaGetErrorString(error); 
      std::cout<<lastError<<std::endl;

    #endif

      return 0;

    }



    int _prec_mvm ( mode_vector<FLOAT_TYPE,int> input,
                    mode_vector<FLOAT_TYPE,int> output ) const
    {
#ifdef USE_PRECONDITIONER

      volume_mvm <FLOAT_TYPE, 1>
      <<<volume_gridSIZE, volume_blockSIZE>>>
      ( order, d_prec_matrix, input, output ); 

    #if 0

      cudaError_t error = cudaGetLastError();
      std::string lastError = cudaGetErrorString(error); 
      std::cout<<lastError<<std::endl;

    #endif

#endif
      return 0;
    }



};



template<typename FLOAT_TYPE>
void sipg_sem_2d<FLOAT_TYPE>::compute_rhs
     ( FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE) )
{

  const int N = order;

  const int dim = mesh.dim();
  const int noe = mesh.noe();

  host_mode_vector<FLOAT_TYPE,int> rhs(dim*dim, order+1);

  std::vector<FLOAT_TYPE> x = mesh.x_coord();
  std::vector<FLOAT_TYPE> y = mesh.y_coord();

  std::vector<FLOAT_TYPE> nodes(N+1), weights(N+1), w(N+1);

  for (int i = 0; i <= N; ++i)
  {
    nodes[i] = (qt.node(i)*.5 + .5)/dim;
    weights[i] = qt.weight(i);
    w[i] = qt.weight(i)*.5/dim;
  }

  for (int i1 = 0; i1 <= N; ++i1)
    for (int i2 = 0; i2 <= N; ++i2)
      for (int e = 0; e < noe; ++e)
        rhs(i1,i2,e) = w[i1]*w[i2]
                         *f( x[e] + nodes[i1],
                             y[e] + nodes[i2] );

  for ( int xx = 0; xx < dim; ++xx )
  {
 
    const int eD = (dim - 1) * dim + xx; //DOWN
    const FLOAT_TYPE xeD = mesh.x_coord()[eD];
    const FLOAT_TYPE yeD = mesh.y_coord()[eD];
 
    const int eU = xx; //UP
    const FLOAT_TYPE xeU = mesh.x_coord()[eU];
    const FLOAT_TYPE yeU = mesh.y_coord()[eU];

    const int eL = xx*dim; //LEFT
    const FLOAT_TYPE xeL = mesh.x_coord()[eL];
    const FLOAT_TYPE yeL = mesh.y_coord()[eL];

    const int eR = xx*dim + dim-1; //RIGHT 
    const FLOAT_TYPE xeR = mesh.x_coord()[eR];
    const FLOAT_TYPE yeR = mesh.y_coord()[eR];
  
    for (int i=0; i<= N; ++i)
    {

      rhs(i,0,eD)  += pen*weights[i]*u_ex(xeD + nodes[i], yeD + nodes[0]);
      rhs(i,N,eU)  += pen*weights[i]*u_ex(xeU + nodes[i], yeU + nodes[N]);
      rhs(0,i,eL)  += pen*weights[i]*u_ex(xeL + nodes[0], yeL + nodes[i]);
      rhs(N,i,eR)  += pen*weights[i]*u_ex(xeR + nodes[N], yeR + nodes[i]);

      for (int j=0; j<= N; ++j)
      {
        rhs(i,j,eD) -= -1*weights[i]*basis.d_phi(j,0)*u_ex(xeD + nodes[i], yeD + nodes[0]);
        rhs(i,j,eU) -=    weights[i]*basis.d_phi(j,N)*u_ex(xeU + nodes[i], yeU + nodes[N]);
        rhs(j,i,eL) -= -1*weights[i]*basis.d_phi(j,0)*u_ex(xeL + nodes[0], yeL + nodes[i]);
        rhs(j,i,eR) -=    weights[i]*basis.d_phi(j,N)*u_ex(xeR + nodes[N], yeR + nodes[i]);
      }

    }
  } 


  copy(rhs, d_rhs);

}





template<typename FLOAT_TYPE>
void sipg_sem_2d<FLOAT_TYPE>::err_norms
     ( host_mode_vector<FLOAT_TYPE,int> & u, 
       FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*dx_u_ex)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*dy_u_ex)(FLOAT_TYPE, FLOAT_TYPE) )
{

  const int dim = mesh.dim(); // number of element on an edge
  const int noe = mesh.noe(); // total number of element

  L2_err = 0;
  H1_err = 0;
  max_err = -1;

  std::vector<FLOAT_TYPE> nodes(order+1), weights(order+1);

  for (int i = 0; i <= order; ++i)
  {
    nodes[i] = (qt.node(i)*.5 + .5)/dim;
    weights[i] = qt.weight(i)*.5/dim;
  }


  FLOAT_TYPE dx_square_err(0), dy_square_err(0), L2_square_err(0);


  for (int e = 0; e < noe; ++e)
    for (int p = 0; p <= order; ++p)     // node index on x-axis 
      for (int q = 0; q <= order; ++q)   // node index on y-axis 
      {

        FLOAT_TYPE dx_u(0), dy_u(0);
        FLOAT_TYPE weight2d = weights[p]*weights[q];

        for (int i = 0; i <= order; ++i)   // mode index on x-axis
          for (int j = 0; j <= order; ++j) // mode index on y-axis
          {
            dx_u += u(i,j,e)*2*dim*basis.d_phi(i,p)*basis.phi(j,q); 
            dy_u += u(i,j,e)*2*dim*basis.phi(i,p)*basis.d_phi(j,q); 
          }

        FLOAT_TYPE xe = mesh.x_coord()[e];
        FLOAT_TYPE ye = mesh.y_coord()[e];
       
        FLOAT_TYPE err =  std::fabs( u(p,q,e) - u_ex(xe + nodes[p], ye + nodes[q]) );
        if ( err > max_err ) max_err = err;
        L2_square_err += weight2d*std::pow( err ,2);
        dx_square_err += weight2d*std::pow( dx_u - dx_u_ex(xe + nodes[p], ye + nodes[q]) ,2);
        dy_square_err += weight2d*std::pow( dy_u - dy_u_ex(xe + nodes[p], ye + nodes[q]) ,2); 

      }

  H1_err = L2_square_err + dx_square_err + dy_square_err;

  L2_err = std::sqrt(L2_square_err);
  H1_err = std::sqrt(H1_err);

}


#include<iomanip>

template<typename FLOAT_TYPE>
void sipg_sem_2d<FLOAT_TYPE>::print_result()
{  

  const int noe = mesh.noe();
  const int dim = mesh.dim();

  std::vector<FLOAT_TYPE> nodes(order+1);
  for (int i = 0; i <= order; ++i)
    nodes[i] = (qt.node(i)*.5 + .5)/dim;

  for (int e = 0; e < noe; ++e)
  {

    FLOAT_TYPE  xe = mesh.x_coord()[e];
    FLOAT_TYPE  ye = mesh.y_coord()[e];

    for (int i = 0; i <= order; ++i)
      for (int j = 0; j <= order; ++j)
        std::cerr<<std::setw(10)<<xe + nodes[i]<<"\t"<<std::setw(10)<<ye + nodes[j]<<"\t"
                 <<std::setw(10)<<solution(i,j,e)
//               <<"\t"<<std::setw(10)<<u_ex(xe +nodes[i],ye + nodes[j])
                 <<std::endl;

    std::cerr<<std::endl;

  }

}


#endif

