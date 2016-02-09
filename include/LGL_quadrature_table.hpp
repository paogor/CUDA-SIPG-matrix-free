#ifndef __LGL_QUADRATURE_TABLE_HPP__
#define __LGL_QUADRATURE_TABLE_HPP__

#include<cmath>
#include<limits>

#include<vector>

#include<legendre_poly.hpp>


/**
This class provides the Legendre-Gauss-Lobatto nodes and weights.
\warning Nodes are ordered from -1 to +1.
*/
template<typename FLOAT_TYPE>
class LGL_quadrature_table
{

  private:

    std::vector<FLOAT_TYPE> _nodes;
    std::vector<FLOAT_TYPE> _weights;
    int _number_of_points;

    FLOAT_TYPE sigma (int degree, int k) const;

  public:

    LGL_quadrature_table (int _number_of_points);

    /** \returns number of quadrature points */
    inline int nop() const
    {
      return _number_of_points;
    }

    /** \param   i node number
        \returns i-th quadrature node */
    inline const FLOAT_TYPE & node(int i) const
    {
      return _nodes[i];
    }

    /** \param   i weight number
        \returns i-th quadrature weight */
    inline const FLOAT_TYPE & weight(int i) const
    {
      return _weights[i];
    }

    /** \return const pointer to the array of nodes */
    inline const FLOAT_TYPE * nodes_array() const
    {
      return _nodes.data();
    }


    /** \return const pointer to the array of weights */
    inline const FLOAT_TYPE * weights_array() const
    {
      return _weights.data();
    }


};


/**
approximation of the zeros of ln. lether (1978)
*/
template<typename FLOAT_TYPE>
FLOAT_TYPE LGL_quadrature_table<FLOAT_TYPE>::sigma (int N, int k) const
{
  const FLOAT_TYPE PI = 3.14159265359;
  const FLOAT_TYPE tetha = (4*k - 1) * PI / FLOAT_TYPE(4*N + 2); 

  return std::cos(tetha) * ( 1. - (N-1.)/(8.*N*N*N) - 
                       ( 39. - ( 28. / (std::sin(tetha)*std::sin(tetha)) ) ) / (384.*N*N*N*N) );
}



template<typename FLOAT_TYPE>
LGL_quadrature_table<FLOAT_TYPE>::LGL_quadrature_table (int number_of_points)
{

  assert(number_of_points > 1);
  
  _number_of_points = number_of_points;
  int degree = _number_of_points - 1;

  _nodes.resize(_number_of_points);
  _weights.resize(_number_of_points);

  const FLOAT_TYPE EPS = std::numeric_limits<FLOAT_TYPE>::epsilon(); 
/*

  std::numeric_limits<T>::epsilon()
  Returns the machine epsilon, that is, the difference between 1.0 and
  the next value representable by the floating-point type T.  

  The values of nodes are in the range of [-1.0 : 1.0], so I use this
  number to determine convergence of the Newton method implemented in
  the inner loop.  

*/

  using namespace legendre_poly;

  _nodes[0]      = -1;     // endpoints are
  _nodes[degree] =  1;     // included a priori

  int L;

  if (degree%2 == 0)
  {
    _nodes[degree/2] = 0;
    L = degree/2;
  } else L = degree/2 + 1; 


  for(int i = 1; i < L; ++i)
  {
   
    FLOAT_TYPE x0;
    FLOAT_TYPE x1 = .5*(sigma(degree, i) + sigma(degree, i+1));
    do
    {

      x0 = x1;
      x1 = x0 - ( (1-x0*x0) * d_leg_p(degree,x0) /
                  ( 2*x0*d_leg_p(degree,x0) - degree*(degree+1)*leg_p(degree,x0) ) );
    }
    while (fabs(x0 - x1) > EPS );

    _nodes[i]          = -1 * x1;
    _nodes[degree - i] =      x1;

  }


  for (int i = 0; i < _number_of_points; ++i)
    _weights[i] = 2. / ( degree*(degree+1)*std::pow(leg_p(degree,_nodes[i]),2) );

}





#endif

