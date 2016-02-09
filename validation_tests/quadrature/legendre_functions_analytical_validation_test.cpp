#include<iostream>

#include<cmath>
#include<cassert>

#include<legendre_poly.hpp>
#include<LGL_quadrature_table.hpp>


// Hard coded Legendre polynomials and derivatives.
template<typename T> T   analytic_legendreP(int N, T z);
template<typename T> T  analytic_DlegendreP(int N, T z);
template<typename T> T analytic_D2legendreP(int N, T z);

template<typename T> T legendre_max_error_in_GLL_nodes (int D_order);



int main()
{

  std::cout<<"Max error for Legendre polynomials:\t\t\t\t"
           <<legendre_max_error_in_GLL_nodes<double>(0)<<std::endl;

  std::cout<<"Max error for first-derivative of Legendre polynomials:\t"
           <<legendre_max_error_in_GLL_nodes<double>(1)<<std::endl;

  std::cout<<"Max error for second-derivative of Legendre polynomials:\t"
           <<legendre_max_error_in_GLL_nodes<double>(2)<<std::endl;

  return 0;
} 




/**
  Max error betweens 'three-term recurrence' and
  'analytic formulation' at GLL nodes;
*/

template<typename T>
T legendre_max_error_in_GLL_nodes (int D_order)
{
  
  assert( (D_order>-1) && (D_order<3) );

  using std::max;
  using std::fabs;

  const int N = 10;       //< max Legendre-order 

  T max_err = 0.;


  for (int n = 0; n <= N; ++n)
  {

    LGL_quadrature_table<T> q(n+1);


    for (int i = 0; i <= n; ++i)
    {

      T z = q.node(i);
      T err;

      switch(D_order)
      {
        case 0:
         err = fabs(legendre_poly::leg_p(n,z) - analytic_legendreP(n, z));
         break;
        case 1:
         err = fabs(legendre_poly::d_leg_p(n, z) - analytic_DlegendreP(n, z));
         break;
        case 2:
         err = fabs(legendre_poly::d2_leg_p(n, z) - analytic_D2legendreP(n, z));
         break;
      }
        
      max_err = max(err, max_err);
  
     std::cerr<<n<<"\t"<<i<<"\t"<<err<<std::endl;  

    }

  }

  return max_err;

}




/**
  Max error betweens 'three-term recurrence' and 'analytic formulation'.
*/
template<typename T>
T legendre_max_error (int D_order, int num_of_points = 10e3)
{
  
  assert( (D_order>-1) && (D_order<3) && (num_of_points>0) );

  using std::max;
  using std::fabs;

  const int N = 10;       //< max Legendre-order 
  const T dz = T(2)/T(num_of_points); 

  T max_err = 0.;


  for (int n = 0; n <= N; ++n)
    for (int i = 0; i <= num_of_points; ++i)
      {
        T z = -1. + i*dz;
        T err;

        switch(D_order)
        {
          case 0:
           err = fabs(legendreP(n, z) - analytic_legendreP(n, z));
           break;
          case 1:
           err = fabs(DlegendreP(n, z) - analytic_DlegendreP(n, z));
           break;
          case 2:
           err = fabs(D2legendreP(n, z) - analytic_D2legendreP(n, z));
           break;
        }
        
        max_err = max(err, max_err);
      }

  return max_err;

}



/**
  Legendre polynomials from Wikipedia page.
  FOR VALIDATION TESTS
*/
template<typename T>
T analytic_legendreP(int N, T z)
{

  assert ( N>-1 && N<11 );

  using std::pow;

  switch(N)
  {
    case 0:
      return 1;
    case 1:
      return z;
    case 2:
      return .5 * (3*z*z - 1);
    case 3:
      return .5 * (5*z*z*z - 3*z);
    case 4:
      return .125 * (35*z*z*z*z - 30*z*z + 3);
    case 5:
      return .125 * (63*z*z*z*z*z - 70*z*z*z + 15*z);
        
    case 6:
      return (231*pow(z,6) - 315*pow(z,4) + 105*z*z - 5) / 16.;
    case 7:
      return (429*pow(z,7) - 693*pow(z,5) + 315*z*z*z - 35*z) / 16.;

    case 8:
      return ( 6435*pow(z,8) - 12012*pow(z,6) +
               6930*pow(z,4) - 1260*z*z + 35 ) / 128;
    case 9: 
      return ( 12155*pow(z,9) - 25740*pow(z,7) +
               18018*pow(z,5) - 4620*z*z*z + 315*z ) / 128.;

    case 10:
      return ( 46189*pow(z,10) - 109395*pow(z,8) +
               90090*pow(z,6) - 30030*pow(z,4) + 3465*z*z - 63 ) / 256.;    
  }

}

/**
  Legendre polynomial first-derivatives.
  FOR VALIDATION TESTS
*/
template<typename T>
T analytic_DlegendreP(int N, T z)
{

  assert ( N>-1 && N<11 );

  using std::pow;

  switch(N)
  {
    case 0:
      return 0;
    case 1:
      return 1;
    case 2:
      return 3*z;
    case 3:
      return .5 * (15*z*z - 3);
    case 4:
      return .125 * (35*4*z*z*z - 60*z);
    case 5:
      return .125 * (63*5*z*z*z*z - 210*z*z + 15);
        
    case 6:
      return (231*6*pow(z,5) - 315*4*pow(z,3) + 210*z) / 16.;
    case 7:
      return (429*7*pow(z,6) - 693*5*pow(z,4) + 315*3*z*z - 35) / 16.;

    case 8:
      return ( 6435*8*pow(z,7) - 12012*6*pow(z,5) +
               6930*4*pow(z,3) - 1260*2*z ) / 128;
    case 9: 
      return ( 12155*9*pow(z,8) - 25740*7*pow(z,6) +
               18018*5*pow(z,4) - 4620*3*z*z + 315 ) / 128.;

    case 10:
      return ( 461890*pow(z,9) - 109395*8*pow(z,7) +
               90090*6*pow(z,5) - 30030*4*pow(z,3) + 3465*2*z ) / 256.;    
  }

}

/**
  Legendre polynomial second-derivatives.
  FOR VALIDATION TESTS
*/
template<typename T>
T analytic_D2legendreP(int N, T z)
{

  assert ( N>-1 && N<11 );

  using std::pow;

  switch(N)
  {
    case 0:
      return 0;
    case 1:
      return 0;
    case 2:
      return 3;
    case 3:
      return 15*z;
    case 4:
      return .125 * (35*12*z*z - 60);
    case 5:
      return .125 * (63*20*z*z*z - 420*z );
        
    case 6:
      return (231*30*pow(z,4) - 315*12*z*z + 210) / 16.;
    case 7:
      return (429*42*pow(z,5) - 693*20*pow(z,3) + 315*6*z) / 16.;

    case 8:
      return ( 6435*56*pow(z,6) - 12012*30*pow(z,4) +
               6930*12*pow(z,2) - 1260*2 ) / 128.;
    case 9: 
      return ( 12155*72*pow(z,7) - 25740*42*pow(z,5) +
               18018*20*pow(z,3) - 4620*6*z ) / 128.;

    case 10:
      return ( 461890*9*pow(z,8) - 109395*56*pow(z,6) +
               90090*30*pow(z,4) - 30030*12*pow(z,2) + 3465*2 ) / 256.;    
  }

}




