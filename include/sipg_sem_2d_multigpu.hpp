#ifndef __SIPG_SEM_2D_MULTIGPU_HPP__
#define __SIPG_SEM_2D_MULTIGPU_HPP__


#include<sem_function.hpp>
#include<abs_mvm_multigpu.hpp>
#include<mode_vector.hpp>
#include<LGL_quadrature_table.hpp>
#include<sipg_sem_2d_gpu_kernels_multigpu.hpp>
#include<build_square_mesh.hpp>

#include<dotproduct_multigpu.hpp>
#include<halo_exchange.h>

#include<conjugate_gradient_multigpu.hpp>

/**
  This class solves the Poisson problem with Dirichlet border condition
  using the Symmetric Interior Penalty Galerkin method with Spectral Element
  Method basis on **MULTIPLE GPU**.

  All the work is done by the constructor. It generates the right hand side,
  initializes the device memory, calls the Conjugate Gradient solver.
*/
template<typename FLOAT_TYPE>
class sipg_sem_2d_multigpu : public abs_mvm_multigpu<FLOAT_TYPE>
{

  private:

    int order;
    square_mesh_multigpu<FLOAT_TYPE> mesh;
    LGL_quadrature_table<FLOAT_TYPE> qt;
    sem_function<FLOAT_TYPE> basis;

    mode_vector<FLOAT_TYPE,int> output;

    FLOAT_TYPE pen; 


    void compute_rhs( FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
                      FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE) );

    void err_norms( host_mode_vector<FLOAT_TYPE,int> & u, 
                    FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
                    FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE),
                    FLOAT_TYPE (*dx_u_ex)(FLOAT_TYPE, FLOAT_TYPE),
                    FLOAT_TYPE (*dy_u_ex)(FLOAT_TYPE, FLOAT_TYPE) );


    // +++++++++++++++++++++++++++ GCL stuff +++++++++++++++++++++++++++ 

    typedef GCL::layout_map<0,1,2> layoutmap;
    typedef GCL::halo_exchange_generic<GCL::layout_map<0,1,2>, 3, GCL::gcl_gpu, GCL::version_manual> pattern_type;

    pattern_type he;
    GCL::field_on_the_fly<FLOAT_TYPE, layoutmap, pattern_type::traits> field_gpu;

  public:

    FLOAT_TYPE max_err;
    FLOAT_TYPE L2_err;
    FLOAT_TYPE H1_err;

    mode_vector<FLOAT_TYPE,int> d_u;
    mode_vector<FLOAT_TYPE,int> d_rhs;

    host_mode_vector<FLOAT_TYPE,int> solution;

    dotproduct_multigpu<FLOAT_TYPE> dot_product;

    sipg_sem_2d_multigpu ( MPI_Comm CartComm,
                           int _order, 
                           const square_mesh_multigpu<FLOAT_TYPE> & _mesh,
                           FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
                           FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE), 
                           FLOAT_TYPE (*dx_u_ex)(FLOAT_TYPE, FLOAT_TYPE), 
                           FLOAT_TYPE (*dy_u_ex)(FLOAT_TYPE, FLOAT_TYPE) )
     : qt(_order+1),
       basis(_order),
       pen(100*_order*_order),
       output(_mesh.noe(), _order+1),

       mesh(_mesh),
       dot_product(mesh.device_info, _order),

       he(typename pattern_type::grid_type::period_type(0, 0, 0), CartComm)
    {

      const int noe = _mesh.noe();
      order = _order;
      // initialize
      load_Dphi_table<FLOAT_TYPE>(order);
      load_lgl_quadrature_table<FLOAT_TYPE>(order);

      host_mode_vector<FLOAT_TYPE,int> h_xx(noe, order+1);

      setup_GCL();

#if 0

  int pid;
  MPI_Comm_rank(MPI_COMM_WORLD, &pid);

  const int n = mesh.local_dim();
  for(int m1 = 0; m1< order+1; ++m1 )
    for(int m2 = 0; m2< order+1; ++m2 )
      for(int i = 0; i < n; ++i) // row
        for(int j = 0; j < n; ++j) // col
          h_xx(m1, m2, n*i + j) = 1; //FLOAT_TYPE(m1+1)/(m2+1);// (pid*10000) + (m1*1000) + (m2*100) + (n*i + j);

   //   copy(h_xx, output);

#endif


      compute_rhs(f, u_ex);
      copy(h_xx, d_u);

      int it = conjugate_gradient_multigpu(*(this), d_u, d_rhs);
 //     _mvm(d_rhs);
      // copy back the solution 
      copy(d_u, solution);

      err_norms(solution, f, u_ex, dx_u_ex, dy_u_ex);

    }

    ~sipg_sem_2d_multigpu ()
    {
      d_rhs.free();
      d_u.free();
      output.free();
    }



    void print_result();
           

    int _mvm ( mode_vector<FLOAT_TYPE,int> input)
    {
 
      const int noe = input.get_noe();
      const int blockD = 128;

      volume<FLOAT_TYPE>
      <<<dim3( (noe + blockD - 1)/blockD , order+1, order+1), blockD>>>
      ( order, input, output ); 

      const int dimx = mesh.device_info.get_dimx();
      const int dimy = mesh.device_info.get_dimy();
      const int blockDx = 32;
      const int blockDy = 4;

      local_flux_term6a<FLOAT_TYPE>
      <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
          dim3( blockDx, blockDy, 1 ) >>>
      ( order, mesh.device_info, input, output );

      local_flux_term6b<FLOAT_TYPE>
      <<< dim3( (dimx + blockDx - 1)/blockDx, (dimy + blockDy - 1)/blockDy, 1 ) ,
          dim3( blockDx, blockDy, 1 ) >>>
      ( order, mesh.device_info, input, output );

      // use GCL in order to exchange output HALOS  
      he.pack(field_gpu);
      he.exchange();
      he.unpack(field_gpu);

      #if 0

       cudaError_t error = cudaGetLastError();
       std::string lastError = cudaGetErrorString(error); 
       std::cout<<lastError<<std::endl;

      #endif

      return 0;

    }

    mode_vector<FLOAT_TYPE,int> _mvm_output()
    {
      return output;
    }

 
    FLOAT_TYPE _dot_product ( mode_vector<FLOAT_TYPE,int> a,
                              mode_vector<FLOAT_TYPE,int> b )
    {
      return dot_product(a, b);
    }


    void setup_GCL()
    {

      const int n = mesh.local_dim();

      GCL::array<GCL::halo_descriptor,3> halo_dsc;
      halo_dsc[1] = GCL::halo_descriptor(1, 1, 1, n-2, n); // row 
      halo_dsc[0] = GCL::halo_descriptor(1, 1, 1, n-2, n); // col
      halo_dsc[2] = GCL::halo_descriptor(0, 0, 0, (order+1)*(order+1), (order+1)*(order+1));

      he.setup( 2,
                GCL::field_on_the_fly<FLOAT_TYPE,layoutmap, pattern_type::traits>(NULL,halo_dsc),
                sizeof(FLOAT_TYPE) );

      GCL::field_on_the_fly<FLOAT_TYPE, layoutmap, pattern_type::traits>
           _field_gpu(reinterpret_cast<FLOAT_TYPE*>(output.data()), halo_dsc);

      field_gpu = _field_gpu; 

    }

};



template<typename FLOAT_TYPE>
void sipg_sem_2d_multigpu<FLOAT_TYPE>::compute_rhs
     ( FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE) )
{

  const int N = order;

  const int dim = mesh.dim();
  const int noe = mesh.noe();
  const int local_dim = mesh.local_dim();

  host_mode_vector<FLOAT_TYPE,int> rhs(local_dim*local_dim, order+1);

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

  if( mesh.DOWNborder() )  
    for ( int xx = 0; xx < local_dim; ++xx ) //DOWN
    {
 
      int e = (local_dim - 2) * local_dim + xx;

      for (int i=0; i<= N; ++i)
      {
        rhs(i,0,e)  += pen*weights[i]*u_ex( x[e] + nodes[i], y[e] + nodes[0]);
        for (int j=0; j<= N; ++j)
          rhs(i,j,e) -= -1*weights[i]*basis.d_phi(j,0)*u_ex(x[e] + nodes[i], y[e] + nodes[0]);
      }
    }



  if( mesh.UPborder() ) 
    for ( int xx = 0; xx < local_dim; ++xx ) //UP
    {

      int e = local_dim + xx;

      for (int i=0; i<= N; ++i)
      {
        rhs(i,N,e)  += pen*weights[i]*u_ex(x[e] + nodes[i], y[e] + nodes[N]);
        for (int j=0; j<= N; ++j)
          rhs(i,j,e) -= weights[i]*basis.d_phi(j,N)*u_ex(x[e] + nodes[i], y[e] + nodes[N]);
       }
    }




  if ( mesh.LEFTborder() ) 
    for ( int yy = 0; yy < local_dim; ++yy ) //LEFT
    {

      int e = yy*local_dim + 1;

      for (int i=0; i<= N; ++i)
      {
        rhs(0,i,e)  += pen*weights[i]*u_ex(x[e] + nodes[0], y[e] + nodes[i]);
        for (int j=0; j<= N; ++j)
          rhs(j,i,e) -= -1*weights[i]*basis.d_phi(j,0)*u_ex(x[e] + nodes[0], y[e] + nodes[i]);
      }
    }




  if ( mesh.RIGHTborder() ) 
    for ( int yy = 0; yy < local_dim; ++yy ) //RIGHT
    {

      int e = yy*local_dim + local_dim-2 ;

      for (int i=0; i<= N; ++i)
      {
        rhs(N,i,e)  += pen*weights[i]*u_ex(x[e] + nodes[N], y[e] + nodes[i]);
        for (int j=0; j<= N; ++j)
          rhs(j,i,e) -=    weights[i]*basis.d_phi(j,N)*u_ex(x[e] + nodes[N], y[e] + nodes[i]);
      } 
    }


  copy(rhs, d_rhs);

}





template<typename FLOAT_TYPE>
void sipg_sem_2d_multigpu<FLOAT_TYPE>::err_norms
     ( host_mode_vector<FLOAT_TYPE,int> & u, 
       FLOAT_TYPE (*f)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*u_ex)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*dx_u_ex)(FLOAT_TYPE, FLOAT_TYPE),
       FLOAT_TYPE (*dy_u_ex)(FLOAT_TYPE, FLOAT_TYPE) )
{

  const int dim = mesh.dim(); // number of element on an edge
  const int local_dim = mesh.local_dim(); // number of element on an edge of the tile (include halos)
  const int noe = mesh.noe(); // total number of element on the tile

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


  for (int ii = 1; ii < local_dim-1; ++ii)     // these two loops
    for (int jj = 1; jj < local_dim-1; ++jj)   // exclude the halos
    {
      
      const int e = ii*local_dim + jj;

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

    }

  FLOAT_TYPE H1_square_err = L2_square_err + dx_square_err + dy_square_err;

  MPI_Allreduce(&L2_square_err, &L2_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&H1_square_err, &H1_err, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  L2_err = std::sqrt(L2_err);
  H1_err = std::sqrt(H1_err);

}


#include<iomanip>

template<typename FLOAT_TYPE>
void sipg_sem_2d_multigpu<FLOAT_TYPE>::print_result()
{  
#if 0
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
#endif
}


#endif

