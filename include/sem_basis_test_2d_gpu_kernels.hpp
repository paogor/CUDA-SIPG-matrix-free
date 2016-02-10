#ifndef __SEM_BASIS_TEST_2D_GPU_KERNELS_HPP__
#define __SEM_BASIS_TEST_2D_GPU_KERNELS_HPP__

#include<CUDA_ERRORS.hpp> 
#include<device_const.hpp>
#include<abs_mvm.hpp>
#include<volume_gpu_kernels.hpp>


/** This namespace contains **kernels** which tests SEM basis. 
    These kernels solve a Poisson problem on a single square element
    with Nitsche boundary conditions. ONLY for TESTS VALUES of
    basis function derivatives. 
*/
namespace sem_2d_test_namespace
{

  template<typename FLOAT_TYPE> 
  __global__ void penalty_terms ( int N,
                                  mode_vector<FLOAT_TYPE, int> input, 
                                  mode_vector<FLOAT_TYPE, int> output )
  {

    int idx = blockIdx.x*blockDim.x + threadIdx.x; 

    if ( !(idx < input.get_noe()) ) return;

    const FLOAT_TYPE nu = 10; // check

    FLOAT_TYPE * c_lgl = device_const::lgl_weights<FLOAT_TYPE>(); 

    for (int i = 0; i <= N ; ++i)
    { 
      output(i,0,idx) +=  .5 * c_lgl[i] * nu * input(i,0,idx); 
 
      output(i,N,idx) +=  .5 * c_lgl[i] * nu * input(i,N,idx); 

      output(0,i,idx) +=  .5 * c_lgl[i] * nu * input(0,i,idx); 

      output(N,i,idx) +=  .5 * c_lgl[i] * nu * input(N,i,idx); 
    }

    return;

  }





  template<typename FLOAT_TYPE>
  __global__ void flux_terms( int N,
                              mode_vector<FLOAT_TYPE,int> input,
                              mode_vector<FLOAT_TYPE,int> output )
  {

    int idx = blockIdx.x*blockDim.x + threadIdx.x; 

    if ( !(idx < input.get_noe()) ) return;

    FLOAT_TYPE * c_lgl = device_const::lgl_weights<FLOAT_TYPE>(); // pointer to LGL weights in device constant memory 


    for (int j=0; j <= N; ++j)
      for (int i=0; i <= N; ++i)
      {
        //LEFT
        output(0,j,idx) -= -1*c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[0] * input(i,j,idx);
        output(i,j,idx) -= -1*c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[0] * input(0,j,idx);

        //RIGHT
        output(N,j,idx) -= c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[N] * input(i,j,idx);
        output(i,j,idx) -= c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[N] * input(N,j,idx);

        //DOWN
        output(j,0,idx) -= -1*c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[0] * input(j,i,idx);
        output(j,i,idx) -= -1*c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[0] * input(j,0,idx);

        //UP
        output(j,N,idx) -= c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[N] * input(j,i,idx);
        output(j,i,idx) -= c_lgl[j]*DPHI<FLOAT_TYPE>(N,i)[N] * input(j,N,idx);
      }  


    return;
  }


}






template<typename FLOAT_TYPE>
class sem_2d_test : public abs_mvm<FLOAT_TYPE>
{

  private:

    int degree;

  public:

    __host__ sem_2d_test (int _degree) : degree(_degree) 
    {
      // initialize
      load_Dphi_table<FLOAT_TYPE>(degree);
      load_lgl_quadrature_table<FLOAT_TYPE>(degree);
    }


    int _mvm ( mode_vector<FLOAT_TYPE,int> input,
               mode_vector<FLOAT_TYPE,int> output ) const
    {
 
      const int noe = input.get_noe();
      const int blockD = 32;

      volume<FLOAT_TYPE>
      <<<dim3( (noe + blockD - 1)/blockD , degree+1, degree+1), blockD>>>
      ( degree , input, output ); 
 
      sem_2d_test_namespace::flux_terms<FLOAT_TYPE>
      <<<dim3( (noe + blockD - 1)/blockD , 1, 1 ), blockD>>>
      ( degree , input, output );

      sem_2d_test_namespace::penalty_terms<FLOAT_TYPE>
      <<<dim3( (noe + blockD - 1)/blockD , 1, 1 ), blockD>>>
      ( degree , input, output );


#if 0

      cudaError_t error = cudaGetLastError();
      std::string lastError = cudaGetErrorString(error); 
      std::cout<<lastError<<std::endl;

#endif

      return 0;
    }

};



#endif
