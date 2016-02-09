#ifndef __LEGENDRE_POLY_HPP__
#define __LEGENDRE_POLY_HPP__

#include<cmath>
#include<cassert>
#include<limits>


/** \brief Legendre polynomials and their derivatives */
namespace legendre_poly
{


/**
  \brief Legendre polynomial

  Three-term recurrence relation is used to evaluate polynomial value.
 
  \param N polynomial degree
  \param z evaluation point [-1;1]

  \return Nth-Legendre polynomial at point z

*/

  template<typename FLOAT_TYPE>
  FLOAT_TYPE leg_p(int N, FLOAT_TYPE z)
  {

    assert (N > -1);

    FLOAT_TYPE l0 = 1;

    if (N==0)
      return l0;

    FLOAT_TYPE  l1 = z; //

    for (int n = 1; n < N; ++n)
    {
      // l2 is L_(N+1)(z)
      const FLOAT_TYPE  l2 = ( (2*n+1)*z*l1 - n*l0 ) / (n+1);

      l0=l1;
      l1=l2;
    }

    return l1;

  }



/**
  \brief First-order derivative of Nth-Legendre polynomial.

  From p.95
 
  \param N order of polynomial
  \param z evaluation point [-1;1]

  \return First-order derivative of Nth-Legendre polynomial at point z

*/

  template<typename FLOAT_TYPE>
  FLOAT_TYPE d_leg_p(int N, FLOAT_TYPE z)
  {

    const FLOAT_TYPE EPS = std::numeric_limits<FLOAT_TYPE>::epsilon() * 10;
    assert ( std::fabs(z) < 1 + EPS );
    assert ( N > -1 );


    if (N == 0 ) return 0;

    if ( std::fabs(z + 1) > EPS  && std::fabs(z - 1) > EPS  )  // (3.176d)
 
     return (leg_p(N-1,z)-leg_p(N+1,z)) * N * (N+1) / ((2*N + 1)*(1 - z*z));

    else return .5*N*(N+1)*std::pow(z, N-1); // (3.177a)

  }



/**
  \brief Second-order derivative of Nth-Legendre polynomial.

  pp. 94-95

  \param N order of polynomial
  \param z evaluation point [-1;1]

  \return Second-order derivative of Nth-Legendre polynomial at point z

*/

  template<typename FLOAT_TYPE>
  FLOAT_TYPE d2_leg_p(int N, FLOAT_TYPE z)
  {

    const FLOAT_TYPE EPS = std::numeric_limits<FLOAT_TYPE>::epsilon() * 10;
    assert ( std::fabs(z) < 1 + EPS );
    assert (N > -1);


    if ( std::fabs(z + 1) > EPS  && std::fabs(z - 1) > EPS  )     //  (3.172)

      return ( 2 * z * d_leg_p(N,z) - N * (N+1) * leg_p(N,z) ) / (1 - z*z);

    else return .125 * (N-1) * N * (N+1) * (N+2) * std::pow(z,N); // (3.177b)

  }


} 

#endif

