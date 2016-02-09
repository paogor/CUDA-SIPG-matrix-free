#include<iostream>

#include<mode_matrix_kernels.hpp>


int main()
{
  const int noe = 3;
  const int order = 3;
  host_mode_matrix<double,int> m(noe, order);
  const int nzpr = m.get_nzpr();


  for (int e = 0; e < noe; e++)
  {

    for (int i1 = 0; i1 <= order; i1++)
      for (int i2 = 0; i2 <= order; i2++)
      {
        for (int r_idx= 0; r_idx < nzpr; r_idx++)
          std::cout<<m(i1,i2, r_idx,e)<<" ";

        std::cout<<std::endl;
      }

    std::cout<<std::endl;

   }

  mode_matrix<double, int> d_m; 
  d_m = m;

  mode_vector<double, int> in(noe, order+1);
  mode_vector<double, int> out(noe, order+1);

  volume_mvm<double>
  <<<dim3(noe, order+1, order+1), 1>>>
  (order, d_m, in, out);


  cudaError_t error = cudaGetLastError();
  std::string lastError = cudaGetErrorString(error); 
  std::cout<<lastError<<std::endl;




  return 0;
}
