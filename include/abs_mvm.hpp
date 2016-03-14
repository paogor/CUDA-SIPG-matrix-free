#ifndef __ABS_MVM__
#define __ABS_MVM__

#include<mode_vector.hpp>
/**
  This abstract class provides an interface for matrix-vector product methods
  used by iterative solvers.
  A define-problem class needs to implement this method. 
*/
template<typename FLOAT_TYPE>
class abs_mvm 
{
  public:
    virtual int _mvm ( mode_vector<FLOAT_TYPE,int>,
                       mode_vector<FLOAT_TYPE,int> ) const = 0;

    virtual int _prec_mvm ( mode_vector<FLOAT_TYPE,int>,
                            mode_vector<FLOAT_TYPE,int> ) const = 0;
};


#endif
