#ifndef __ABS_MULTIGPU_MVM__
#define __ABS_MULTIGPU_MVM__

#include<mode_vector.hpp>
/**
  This abstract class provides an interface for matrix-vector product methods
  used by iterative solvers on **MULTIGPU**. 
  A define-problem **MULTIGPU** class needs to implement this method. 
*/
template<typename FLOAT_TYPE>
class abs_mvm_multigpu 
{

  public:
    /** This method performs the matrix vector product */
    virtual int _mvm ( mode_vector<FLOAT_TYPE,int> input ) = 0;
    /** This method returns the result 
        of the last matrix vector product */
    virtual mode_vector<FLOAT_TYPE,int>& _mvm_output() = 0;

    virtual FLOAT_TYPE _dot_product ( mode_vector<FLOAT_TYPE,int>, mode_vector<FLOAT_TYPE,int> ) = 0;

    virtual int _prec_mvm ( mode_vector<FLOAT_TYPE,int>,
                            mode_vector<FLOAT_TYPE,int> ) const = 0;

};


#endif
