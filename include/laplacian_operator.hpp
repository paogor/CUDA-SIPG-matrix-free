#ifndef __LAPLACIAN_MATRIX_HPP__
#define __LAPLACIAN_MATRIX_HPP__


#include<sem_function.hpp>

#define SEM_FUNC sem_function<FLOAT_TYPE>

template<typename FLOAT_TYPE>
class laplace_1d : public SEM_FUNC
{

  public:

  laplace_1d(int degree) : SEM_FUNC(degree) {};

  FLOAT_TYPE operator() (int i, int j)
  {
   
    FLOAT_TYPE r = 0; 

    for (int z = 0; z < SEM_FUNC::_qt.nop(); ++z)
      r +=   SEM_FUNC::_qt.weight(z)
           * SEM_FUNC::d_phi(i,z)
           * SEM_FUNC::d_phi(j,z) ;
     
    return r;

  }

};  


template<typename FLOAT_TYPE>
class laplace_2d : public SEM_FUNC
{

  public:

  laplace_2d(int degree) : SEM_FUNC(degree) {};


  FLOAT_TYPE operator() (int i1, int i2, int j1,  int j2)
  {
    
    FLOAT_TYPE result = 0;

    if (i2 == j2)
      for (int k = 0; k < SEM_FUNC::_qt.nop(); ++k)
        result +=  SEM_FUNC::_qt.weight(k)
                  *SEM_FUNC::_qt.weight(i2)
                  *SEM_FUNC::d_phi(i1, k)
                  *SEM_FUNC::d_phi(j1, k);

    if (i1 == j1)
      for (int k = 0; k < SEM_FUNC::_qt.nop(); ++k)
        result +=  SEM_FUNC::_qt.weight(k)
                  *SEM_FUNC::_qt.weight(i1)
                  *SEM_FUNC::d_phi(i2, k)
                  *SEM_FUNC::d_phi(j2, k);

    return result;

  }

};  

#endif 


