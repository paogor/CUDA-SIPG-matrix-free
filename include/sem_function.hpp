#ifndef __SEM_FUNCTION_HPP__
#define __SEM_FUNCTION_HPP__

#include<cmath>
#include<limits>
#include<cassert>

#include<vector>

#include<legendre_poly.hpp>
#include<LGL_quadrature_table.hpp>

/**
This class provide the values of SEM basis and its
derivatives on LGL quadrature nodes.
*/
template<typename FLOAT_TYPE>
class sem_function
{

  private:

    int _degree;          //< base function degree 
    int _number_of_nodes; //< # of quadrature nodes [_degree + 1]
    int _number_of_modes; //< # of basis function [_degree + 1]


    std::vector<FLOAT_TYPE> _leg_p; ///< Legendre poly values on LGL nodes 

  public:

    LGL_quadrature_table<FLOAT_TYPE> _qt;

    sem_function(int degree);

    inline int non() const
    {
      return _number_of_nodes;
    }

    inline int nom() const
    {
      return _number_of_modes;
    }

    /** This method returns the value of the mode-th
        SEM base at the node_index-th LGL nodes */
    FLOAT_TYPE   phi(int mode, int node_index) const;

    /** This method return the value of the mode-th
        SEM base derivative at the node_index-th LGL nodes */
     FLOAT_TYPE d_phi(int mode, int node_index) const;
  
};


template<typename FLOAT_TYPE>
sem_function<FLOAT_TYPE>::sem_function(int degree)

: _degree(degree),
  _qt(degree+1),
  _leg_p(degree+1),
  _number_of_nodes(degree+1),
  _number_of_modes(degree+1)

{

  using legendre_poly::leg_p;

  for (int i=0; i <_number_of_nodes; ++i)
    _leg_p[i] = leg_p(_degree, _qt.node(i));

}


template<typename FLOAT_TYPE>
FLOAT_TYPE sem_function<FLOAT_TYPE>::phi(int mode, int node_index) const
{
  assert(mode < nom());
  assert(node_index < non());

  return ( mode == node_index ) ? 1 : 0 ;
}


template<typename FLOAT_TYPE>
FLOAT_TYPE sem_function<FLOAT_TYPE>::d_phi(int mode, int node_index) const
{
  assert(mode < nom());
  assert(node_index < non());

  if ( mode == node_index )
  {

    if ( node_index == 0 )
      return -.25*(_degree+1)*_degree; 

    if ( node_index == _degree )
      return  .25*(_degree+1)*_degree;

    return 0;

  }

  else
 
    return _leg_p[node_index]
            /  (  _leg_p[mode] * (_qt.node(node_index) - _qt.node(mode))  );

}


#endif
