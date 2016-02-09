#include<iostream>
#include<cmath>


#include<LGL_quadrature_table.hpp>


template<typename FLOAT_TYPE>
FLOAT_TYPE monomial (int degree, FLOAT_TYPE x)
{
    return std::pow(x,degree); 
}

template<typename FLOAT_TYPE>
FLOAT_TYPE int_monomial (int degree, FLOAT_TYPE a, FLOAT_TYPE b)
{
    return (std::pow(b,degree+1)-std::pow(a,degree+1))/(degree+1); 
} 



template<typename FLOAT_TYPE>
FLOAT_TYPE GLL_int_monomial ( int number_of_quadrature_points,
                              int monomial_degree,
                              FLOAT_TYPE a, FLOAT_TYPE b )
{

  LGL_quadrature_table<FLOAT_TYPE> q(number_of_quadrature_points);

  FLOAT_TYPE result = 0;
  for (int i = 0; i < number_of_quadrature_points; ++i)
    result += .5*(b-a)*q.weight(i)*std::pow( .5*(b-a)*q.node(i) + .5*(a+b), monomial_degree );

  return result;

}




int main()
{

  const int cell_width = 12;

  const int noqp = 9;        // max number of quadrature points
  const int max_degree = 15; // max monomial degree
  const double a = 1.;       // left limit of integration
  const double b = 2.;       // right limit of integration


  // print the fist row of the table

  std::cout<<"-";

  for(int q = 2; q < noqp; ++q)
  {
    std::cout<<" &";
    std::cout.width(cell_width);
    std::cout<<q;
  }

  std::cout<<"\\\\"<<std::endl; 

  // print the following rows of the table,
  // which contains the quadrature errors for a monomial of a given degree
  // varying the number of quadrature points

  for(int degree = 0; degree < max_degree; ++degree)
  {

    std::cout<<degree;

    for(int n = 2; n < noqp; ++n)
    {
      std::cout<<" &";
      std::cout.width(cell_width);
      std::cout<<GLL_int_monomial<double> (n, degree, a, b)
                   - int_monomial<double>    (degree, a, b);

      std::cout.width(cell_width);
      if( degree <= (2*n - 3) ) 
        std::cout<<" \\cellcolor{Gray}";

    }

    std::cout<<"\\\\"<<std::endl; 

  }


  return 0;
} 
