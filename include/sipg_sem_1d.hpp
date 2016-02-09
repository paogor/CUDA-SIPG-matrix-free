#ifndef __SEM_SIPG_SEM_1D_HPP__
#define __SEM_SIPG_SEM_1D_HPP__

#define USE_LAPACK


#include<iomanip>
#include<cassert>


#include<CPU_TIMER.hpp>

#include<sem_function.hpp>

#ifdef USE_LAPACK
  #include<lapack_wrapper.hpp>
#else
  #include <eigen3/Eigen/IterativeLinearSolvers>
  #include <eigen3/Eigen/SparseCore>
  #include <eigen3/Eigen/Dense>
#endif


#define SEM_FUNC sem_function<FLOAT_TYPE>

/**

 This class solves -u" = f on [-1,1].
 It uses the Symmetric Interior Penalty Galerkin method
 with the Spectral Element Method basis functions.

 The constructor does everything.
 The constructor calls the methods.
 The methods fills private data structures.

*/
template<typename FLOAT_TYPE>
class sipg_sem_1d : public sem_function<FLOAT_TYPE>
{

  private:

    int _noe;        //< number of elements
    int _vec_size;   //< size of global vectors (_rhs, _x, _w)
    
    FLOAT_TYPE _pen; //< penalty term

    std::vector<FLOAT_TYPE> _x; //< global nodes coordinates 
    std::vector<FLOAT_TYPE> _w; //< global quadrature weight 

#ifdef USE_LAPACK

    std::vector<FLOAT_TYPE> _A;   //< stiffness matrix
    std::vector<FLOAT_TYPE> _rhs; //< right hand side vector

    std::vector<FLOAT_TYPE> _u; //< solution  

# else 

    Eigen::SparseMatrix<FLOAT_TYPE> _A;   //< stiffness matrix
    Eigen::Matrix<FLOAT_TYPE, Dynamic, 1>  _rhs; //< right hand side vector

    Eigen::Matrix<FLOAT_TYPE, Dynamic, 1> _u; //< solution  

#endif


    std::vector<FLOAT_TYPE> _Al;   //< local stiffness matrix
    std::vector<FLOAT_TYPE> _flux; //< local flux matrix

    FLOAT_TYPE (*_f)(FLOAT_TYPE);    //< pointer pointing to right side function
    FLOAT_TYPE (*_u_ex)(FLOAT_TYPE); //< pointer pointing to exact solution func


    FLOAT_TYPE _max_err;
    FLOAT_TYPE _L2_err;


    void compute_nodes_and_weights();

    void compute_local_matrix();
    void compute_flux_matrix();

    void compute_stiffness_matrix();
    void compute_rhs_vector();

    int solve();

    void compute_err_norms();

    void print_stiffness_matrix(); // call before solve the system

    TIMER stiffness_matrix_generation; 
    TIMER linear_system_solution;
    TIMER total; 

  public:

    sipg_sem_1d( int noe,
                 int degree,
                 FLOAT_TYPE (*_f)(FLOAT_TYPE),
                 FLOAT_TYPE (*_u_ex)(FLOAT_TYPE) );

    
    FLOAT_TYPE max_err() { return _max_err; }
    FLOAT_TYPE L2_err()  { return _L2_err;  }


    void print_local_matrix();
    void print_result();
    void print_err_norms();
    void print_times();

};



template<typename FLOAT_TYPE> 
sipg_sem_1d<FLOAT_TYPE>::sipg_sem_1d( int noe,
                                      int degree,
                                      FLOAT_TYPE (*f)(FLOAT_TYPE),
                                      FLOAT_TYPE (*u_ex)(FLOAT_TYPE) )
  : sem_function<FLOAT_TYPE>::sem_function(degree),
    _noe(noe),
    _f(f),
    _u_ex(u_ex)
{

  assert(noe > 0);

  total.start();

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();

  // penalty term
  _pen = (4*(degree+1)*(degree+1)*noe/2.) + .5;

  _vec_size = nop*noe;

  solve();

  total.stop();

}


/**
 This method computes the global quadrature nodes and weights.
*/
template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::compute_nodes_and_weights()
{

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();

  _x.resize(_vec_size);
  _w.resize(_vec_size);


  for (int e = 0; e < _noe; ++e)
    for (int i = 0; i < nop; ++i)
    {
      _x[e*nop + i] = (SEM_FUNC::_qt.node(i)/_noe)
                      - (FLOAT_TYPE(_noe-1)/FLOAT_TYPE(_noe))
                      + 2.*e*(1./FLOAT_TYPE(_noe)) ;

      _w[e*nop + i] = SEM_FUNC::_qt.weight(i)/_noe;
    }

}



template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::compute_rhs_vector()
{

  compute_nodes_and_weights();

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();
  const int M = nop-1; //< index of the right node
                       //< 0 is the index of the left node
  _rhs.resize(_vec_size);

  for (int i = 0; i < _vec_size; ++i)
    _rhs[i] = _w[i] * _f(_x[i]);

  // this loop add the nitsche border integral term to the first
  // and the last elements 
  // -1 +-----+- ... -+-----+ +1

  for (int i=0; i < nop; ++i)
  {
    _rhs[i]                   +=    _noe*SEM_FUNC::d_phi(i,0)*_u_ex(-1.);
    _rhs[_vec_size - nop + i] += -1*_noe*SEM_FUNC::d_phi(i,M)*_u_ex( 1.);
  }


  _rhs[0]           += _pen*_u_ex(-1.); 
  _rhs[_vec_size-1] += _pen*_u_ex( 1.); 


}


/**
 This method computes local matrix.
 Local matrix includes volume, flux and penalty terms. 
*/
template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::compute_local_matrix()
{

  // local matrices are scaled on [-1/N,1/N]

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();
  const int M = nop-1; //< index of the right node
  _Al.resize(nop*nop);

  for (int i=0; i < nop; ++i)
    for (int j=0; j < nop; ++j)
      for (int z=0; z < nop; ++z)
        _Al[i*nop + j] += SEM_FUNC::_qt.weight(z)
                          * SEM_FUNC::d_phi(i,z)
                          * SEM_FUNC::d_phi(j,z) ;
 
  for (int i=0; i < nop; ++i)
  {

    const FLOAT_TYPE a =  .5*SEM_FUNC::d_phi(i,0);
    _Al[i]     += a;
    _Al[i*nop] += a;

    const FLOAT_TYPE b = -.5*SEM_FUNC::d_phi(i,M);
    _Al[i*nop + (nop-1)] += b;
    _Al[(nop-1)*nop + i] += b;

  }

  for (int i=0; i < nop*nop; ++i)
    _Al[i] *= _noe;

  _Al[0] += _pen;
  _Al[nop*nop - 1] += _pen;

}


/**
  This method computes extra-diagonal flux terms.
  This means the flux between adjacent intervals. 
*/
template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::compute_flux_matrix()
{

  // matrices are scaled on [-1/N,1/N]

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();
  const int M = nop-1; //< index of the right node
  _flux.resize(nop*nop);

  for (int i=0; i < nop; ++i)
  {
    _flux[i*nop]           +=  .5*SEM_FUNC::d_phi(i,M)*_noe;
    _flux[(nop-1)*nop + i] += -.5*SEM_FUNC::d_phi(i,0)*_noe;
  }

}


/**
  This method computes the global stiffness matrix assembling local matrices.
*/
template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::compute_stiffness_matrix()
{

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();
  const int M = nop-1; //< index of the right node

  compute_local_matrix();
  compute_flux_matrix();

  if ( _noe > 1 )
  {

    // add nitsche remaing flux 
    std::vector<FLOAT_TYPE> first = _Al;
    std::vector<FLOAT_TYPE> last  = _Al;

    for (int i=0; i < nop; ++i)
    {
      first[i]     += .5*_noe*SEM_FUNC::d_phi(i,0);
      first[i*nop] += .5*_noe*SEM_FUNC::d_phi(i,0);

      last[i*nop + (nop-1)] += -.5*_noe*SEM_FUNC::d_phi(i,M);
      last[(nop-1)*nop + i] += -.5*_noe*SEM_FUNC::d_phi(i,M);
    }

#ifdef USE_LAPACK

    // + matrix assembly

    _A.resize(_vec_size*_vec_size);

    // ++ diagonal blocks 

    for (int n = 1; n < _noe-1; ++n)
      for (int i=0; i < nop; ++i)
        for (int j=0; j < nop; ++j)
          _A[(i + nop*n)*_vec_size + (j + nop*n)] 
            = _Al[i*nop + j];


    for (int i=0; i < nop; ++i)
      for (int j=0; j < nop; ++j)
      {
        _A[i*_vec_size + j] = first[i*nop + j];
 
        _A[(i+nop*(_noe-1))*_vec_size + (j+nop*(_noe-1))] 
            = last [i*nop + j];
      }


    // ++ extra-diagonal blocks

    for (int n = 0; n < _noe-1; ++n)
      for (int i=0; i < nop; ++i)
        for (int j=0; j < nop; ++j)
        {
          _A[(i+n*nop)*_vec_size + nop*(1+n) + j]   += _flux[i*nop + j];
          _A[(nop*(1+n) + i)*_vec_size + (j+n*nop)] += _flux[j*nop + i];
        }


    for (int n = 1; n < _noe; ++n)
    {
      _A[(n*nop-1)*_vec_size + (n*nop)] += -1.*_pen;  
      _A[n*nop*_vec_size + (n*nop-1)] += -1.*_pen;  
    }

#else


#endif


  }

  else  // _noe == 1

  {

#ifdef USE_LAPACK

    for (int i=0; i < nop; ++i)
    {
      _Al[i]     += .5*_noe*SEM_FUNC::d_phi(i,0);
      _Al[i*nop] += .5*_noe*SEM_FUNC::d_phi(i,0);

      _Al[i*nop + (nop-1)] += -.5*_noe*SEM_FUNC::d_phi(i,M);
      _Al[(nop-1)*nop + i] += -.5*_noe*SEM_FUNC::d_phi(i,M);
    }

    _A = _Al;

#else


#endif

  }


#if 0
  print_stiffness_matrix();
#endif

}



template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::compute_err_norms()
{

  _max_err = -1;

  for (int i = 0; i < _vec_size; ++i)
    _max_err = std::max(std::fabs(_u[i] - _u_ex(_x[i])), _max_err);

  _L2_err = 0;

  for (int i = 0; i < _vec_size; ++i)
  {
    const FLOAT_TYPE err = _u[i] - _u_ex(_x[i]);
    _L2_err += _w[i]*err*err;
  }

  _L2_err = std::sqrt(_L2_err);

}


template<typename FLOAT_TYPE>
int sipg_sem_1d<FLOAT_TYPE>::solve()
{

  stiffness_matrix_generation.start(); // start timer  
  compute_stiffness_matrix();
  stiffness_matrix_generation.stop();  // stop timer


  compute_rhs_vector();



#ifdef USE_LAPACK

  std::vector<int> ipiv(_vec_size,0); // pivot
  _u = _rhs;

  linear_system_solution.start(); // start timer

  int info = lapacke_gesv( LAPACK_ROW_MAJOR,
                           _vec_size,
                           1,
                            _A.data(),
                           _vec_size,
                           ipiv.data(),
                           _u.data(),
                           1                );

#else

  ConjugateGradient<Eigen::SparseMatrix<FLOAT_TYPE> > cg;
  cg.compute(_A);
  _u = cg.solve(_rhs);
 
#endif


  linear_system_solution.stop(); // stop timer 

  compute_err_norms();

  return info;

}



// ****************************** print output ******************************


template<typename float_type>
void sipg_sem_1d<float_type>::print_result()
{

  const int nop = sem_function<float_type>::_qt.nop();

  std::cerr<<std::endl<<std::endl;

  for (int i = 0; i < _vec_size; ++i)
  {
    std::cerr<<std::setw(10)<<_x[i]<<"\t";
    std::cerr<<std::setw(10)<<_u[i]<<"\t";
    std::cerr<<std::setw(10)<<_u_ex(_x[i])<<std::endl;
  }

  std::cerr<<std::endl<<std::endl;



}


template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::print_stiffness_matrix()
{

  std::cout<<std::endl;

  for (int i=0; i < _vec_size; ++i)
  {
    for (int j=0; j < _vec_size; ++j)
      std::cout<<std::setw(10)<<_A[i*_vec_size + j]<<"\t";

    std::cout<<std::endl<<std::endl;    
  }

}


template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::print_local_matrix()
{

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();

  std::cout<<std::endl;

  for (int i=0; i < nop; ++i)
  {
    for (int j=0; j < nop; ++j)
      std::cout<<std::setw(10)<<_Al[i*nop + j]<<"\t";

    std::cout<<std::endl<<std::endl;    
  }

}



template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::print_err_norms()
{

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();
  std::cout<<nop-1<<"\t"<<_noe<<"\t"<<_max_err<<"\t"<<_L2_err<<std::endl;

}



template<typename FLOAT_TYPE>
void sipg_sem_1d<FLOAT_TYPE>::print_times()
{

  const int nop = sem_function<FLOAT_TYPE>::_qt.nop();

  std::cout<<nop-1<<"\t"<<_noe<<"\t";
  std::cout<<stiffness_matrix_generation.elapsed_millisecs()<<"\t"; 
  std::cout<<linear_system_solution.elapsed_millisecs()<<"\t";
  std::cout<<total.elapsed_millisecs()<<std::endl; 

}


#endif 

