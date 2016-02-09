#include<iostream>
#include<cmath>

#include<laplacian_operator.hpp>

int main()
{

  laplace_1d<double> b2(2), b3(3);

  std::cout<<"degree = 2"<<std::endl;

  std::cout<<b2(0,0) - (7./6.)<<std::endl;
  std::cout<<b2(0,1) + (4./3.)<<std::endl;
  std::cout<<b2(0,2) - (1./6.)<<std::endl;
  
  std::cout<<b2(1,0) + (4./3.)<<std::endl;
  std::cout<<b2(1,1) - (8./3.)<<std::endl;
  std::cout<<b2(1,2) + (4./3.)<<std::endl;

  std::cout<<b2(2,0) - (1./6.)<<std::endl;
  std::cout<<b2(2,1) + (4./3.)<<std::endl;
  std::cout<<b2(2,2) - (7./6.)<<std::endl;



  std::cout<<std::endl<<"degree = 3"<<std::endl;

  std::cout<<b3(0,0) - (13./6.)<<std::endl;
  std::cout<<b3(0,1) + (1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;
  std::cout<<b3(0,2) + (-1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;
  std::cout<<b3(0,3) + (1./12.)<<std::endl;
  
  std::cout<<b3(1,0) + (1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;
  std::cout<<b3(1,1) - (800./192.)<<std::endl;
  std::cout<<b3(1,2) + (25./12.)<<std::endl;
  std::cout<<b3(1,3) + (-1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;

  std::cout<<b3(2,0) + (-1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;
  std::cout<<b3(2,1) + (25./12.)<<std::endl;
  std::cout<<b3(2,2) - (800./192.)<<std::endl;
  std::cout<<b3(2,3) + (1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;

  std::cout<<b3(3,0) + (1./12.)<<std::endl;
  std::cout<<b3(3,1) + (-1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;
  std::cout<<b3(3,2) + (1200. + 80*std::pow(5,1.5))/(384.*sqrt(5))<<std::endl;
  std::cout<<b3(3,3) - (13./6.)<<std::endl;


  return 0;
}
