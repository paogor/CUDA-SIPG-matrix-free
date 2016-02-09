#include<iostream>
#include<legendre_poly.hpp>
#include<LGL_quadrature_table.hpp>

/**

  This file print point-wise basis functions and derivatives values. 

  phi and d_phi are written ONLY for plot purposes

*/



double phi(int N, double LGL_node_j, double x)
{

  const double EPS = 1e-10;
  using namespace legendre_poly;

  if ( std::fabs( x - LGL_node_j ) > EPS )
    return -1 * (1 - x*x) * d_leg_p(N,x)
             /( N*(N+1)*( x - LGL_node_j )*leg_p(N, LGL_node_j) );
    else return 1;

}


double d_phi(int N, double LGL_node_j, double x)
{

  const double EPS = 1e-10;
  using namespace legendre_poly;

  if ( std::fabs( x - LGL_node_j ) > EPS )
    return -1 *( (1 - x*x) * d2_leg_p(N,x)  
           - d_leg_p(N,x)*( 2*x + ( (1 - x*x)  / (x - LGL_node_j) )) )
              /( N*(N+1)*leg_p(N, LGL_node_j)*(x-LGL_node_j) );
      else return 0;

}




int main()
{

  const int N = 4;

  LGL_quadrature_table<double> qt(N+1);
  
  double x = -1 + 1e-4;

  while( x < 1 )
  {

    std::cout<<x<<"\t";

    for(int j=0; j<N+1; ++j)
      std::cout<<d_phi(N, qt.node(j), x)<<"\t";

    std::cout<<std::endl;
    x += 1e-2;

  }


  return 0;
}

 
