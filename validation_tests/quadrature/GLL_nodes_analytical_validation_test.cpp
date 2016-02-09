#include<iostream>
#include<vector>
#include<cmath>
#include<LGL_quadrature_table.hpp>


int main()
{


  LGL_quadrature_table<double> q2(3), q3(4), q4(5), q5(6), q6(7);  


  std::cout<<std::endl<<"degree = 2"<<std::endl;
  std::cout<<q2.node(0) + 1.<<std::endl;
  std::cout<<q2.node(1) - 0.<<std::endl;
  std::cout<<q2.node(2) - 1.<<std::endl;


  std::cout<<std::endl<<"degree = 3"<<std::endl;
  std::cout<<q3.node(0) + 1.<<std::endl;
  std::cout<<q3.node(1) + 1./sqrt(5) <<std::endl;
  std::cout<<q3.node(2) - 1./sqrt(5) <<std::endl;
  std::cout<<q3.node(3) - 1.<<std::endl;


  std::cout<<std::endl<<"degree = 4"<<std::endl;
  std::cout<<q4.node(0) + 1. <<std::endl;
  std::cout<<q4.node(1) + sqrt(3)/sqrt(7) <<std::endl;
  std::cout<<q4.node(2) - 0. <<std::endl;
  std::cout<<q4.node(3) - sqrt(3)/sqrt(7) <<std::endl;
  std::cout<<q4.node(4) - 1. <<std::endl;


  std::cout<<std::endl<<"degree = 5"<<std::endl;
  std::cout<<q5.node(0) + 1. <<std::endl;
  std::cout<<q5.node(1) + sqrt(2*sqrt(7) + 7)/sqrt(21) <<std::endl;
  std::cout<<q5.node(2) + sqrt(7 - 2*sqrt(7))/sqrt(21) <<std::endl;
  std::cout<<q5.node(4) - sqrt(2*sqrt(7) + 7)/sqrt(21) <<std::endl;
  std::cout<<q5.node(3) - sqrt(7 - 2*sqrt(7))/sqrt(21) <<std::endl;
  std::cout<<q5.node(5) - 1. <<std::endl;


  std::cout<<std::endl<<"degree = 6"<<std::endl;
  std::cout<<q6.node(0) + 1. <<std::endl;
  std::cout<<q6.node(1) + sqrt(2*sqrt(15) + 15)/sqrt(33) <<std::endl;
  std::cout<<q6.node(2) + sqrt(15 - 2*sqrt(15))/sqrt(33) <<std::endl;
  std::cout<<q6.node(3) - 0. <<std::endl;
  std::cout<<q6.node(5) - sqrt(2*sqrt(15) + 15)/sqrt(33) <<std::endl;
  std::cout<<q6.node(4) - sqrt(15 - 2*sqrt(15))/sqrt(33) <<std::endl;
  std::cout<<q6.node(6) - 1. <<std::endl;



  return 0;
} 
