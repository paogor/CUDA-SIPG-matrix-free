#include<iostream>
#include<sipg_sem_1d.hpp>
 
// ********************************* data *********************************



template<typename FLOAT_TYPE>
FLOAT_TYPE u_ex (FLOAT_TYPE x)
{
  const FLOAT_TYPE  PI = 3.14159265359;
  return std::sin(16*PI*x);
//  return ( 1 - x*x*x*x )*x/5.;
//  return .5 - .5*x*x; 

 const FLOAT_TYPE A = 1;  // u_ex(-1)
 const FLOAT_TYPE B = -6; // u_ex(1)

// return .5*(A+B+1) + .5*(B-A)*x - .5*x*x;

}

template<typename FLOAT_TYPE>
FLOAT_TYPE f (FLOAT_TYPE x)
{
  const FLOAT_TYPE PI = 3.14159265359;
  return 16*16*PI*PI*std::sin(16*PI*x);
//  return 4.*x*x*x;
//  return 1.;
}


template<class PROBLEM>
void L2_trend(int degree)
{

  double L2_err_old;

  for (int noe = 2; noe < 513; noe*=2)
  {  
    PROBLEM p(noe, degree, &f, &u_ex );
    double L2_err = p.L2_err();

    std::cout<<noe<<"\t"<<log(L2_err/L2_err_old)/log(2);
    std::cout<<"\t"<<L2_err<<std::endl;

    L2_err_old = L2_err;
  }

}







int main ()
{

  int noe = 5;
  int degree = 3;

//  for (int noe = 2; noe < 513; ++noe)
  for (int degree = 2; degree < 129; ++degree)
  {
    sipg_sem_1d<double> problem(noe, degree, &f, &u_ex );

    problem.print_err_norms();
  }


  return 0;

}
