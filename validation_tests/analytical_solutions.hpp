#ifndef __ANALYTICAL_SOLUTIONS_HPP__
#define __ANALYTICAL_SOLUTIONS_HPP__

namespace test_func
{

template<typename FLOAT_TYPE>
FLOAT_TYPE u_ex (FLOAT_TYPE x, FLOAT_TYPE y)
{

  const FLOAT_TYPE PI = 3.14159265359;

#if   EXACT_SOLUTION_NO == 1
  return x*y*(y-1)*(x-1)*std::exp(x-y);   // A
#elif EXACT_SOLUTION_NO == 2
  return x*(1-x)*std::exp(3*x)*std::sin(2*PI*y);  // B
#elif EXACT_SOLUTION_NO == 3
  return y*(1-y)*x*x*x; // C
#elif EXACT_SOLUTION_NO == 4
  return x*x*y*y; //D
#elif EXACT_SOLUTION_NO == 5
  return std::cos(8*PI*x) + std::cos(8*PI*y); //E
#else 
    #error unsupported EXACT_SOLUTION_NO
#endif

}


template<typename FLOAT_TYPE>
FLOAT_TYPE f (FLOAT_TYPE x, FLOAT_TYPE y)
{

  const FLOAT_TYPE PI = 3.14159265359;

#if   EXACT_SOLUTION_NO == 1
  return -2*x*(y-1)*(y-2*x+x*y+2)*std::exp(x-y); // A
#elif EXACT_SOLUTION_NO == 2
  return -std::sin(2*PI*y)*std::exp(3*x)*((4-3*x-9*x*x)-4*PI*PI*(x-x*x)); // B
#elif EXACT_SOLUTION_NO == 3
  return 2*x*x*x - 6*x*y*(1-y); // C 
#elif EXACT_SOLUTION_NO == 4
  return -2*y*y -2*x*x; //D
#elif EXACT_SOLUTION_NO == 5
  return 64*PI*PI*std::cos(8*PI*x) + 64*PI*PI*std::cos(8*PI*y); //E
#else 
    #error unsupported EXACT_SOLUTION_NO
#endif

}


template<typename FLOAT_TYPE>
FLOAT_TYPE dx_u_ex (FLOAT_TYPE x, FLOAT_TYPE y)
{
  const FLOAT_TYPE PI = 3.14159265359;

#if   EXACT_SOLUTION_NO == 1
  return (x*x +x -1)*(y-1)*y*std::exp(x-y); //A
#elif EXACT_SOLUTION_NO == 2
  return ( 3*(1-x)*x - x + (1-x) )*std::exp(3*x)*std::sin(2*PI*y); //B
#elif EXACT_SOLUTION_NO == 3
  return y*(1-y)*x*x*3; // C
#elif EXACT_SOLUTION_NO == 4
  return 2*x*y*y; //D
#elif EXACT_SOLUTION_NO == 5
  return -8*PI*std::sin(8*PI*x); //E
#else 
    #error unsupported EXACT_SOLUTION_NO
#endif

}


template<typename FLOAT_TYPE>
FLOAT_TYPE dy_u_ex (FLOAT_TYPE x, FLOAT_TYPE y)
{
  const FLOAT_TYPE PI = 3.14159265359;

#if   EXACT_SOLUTION_NO == 1
  return -1*(y*y - 3*y + 1)*(x-1)*x*std::exp(x-y); //A
#elif EXACT_SOLUTION_NO == 2
  return 2*PI*(1 - x)*x*std::exp(3*x)*std::cos(2*PI*y); //B
#elif EXACT_SOLUTION_NO == 3
  return (1 - 2*y)*x*x*x; // C
#elif EXACT_SOLUTION_NO == 4
  return 2*y*x*x; //D
#elif EXACT_SOLUTION_NO == 5
  return -8*PI*std::sin(8*PI*y); //E
#else 
    #error unsupported EXACT_SOLUTION_NO
#endif

}

} // end of namespace test_func

#endif

