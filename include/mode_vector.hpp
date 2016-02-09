#ifndef __MODE_VECTOR_HPP__
#define __MODE_VECTOR_HPP__

#include<cassert>

#include<CUDA_ERRORS.hpp>


// ===========================================================================


template<typename FLOAT_TYPE, typename INT_TYPE>
class host_mode_vector;

template<typename FLOAT_TYPE, typename INT_TYPE>
class mode_vector;


template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const mode_vector<FLOAT_TYPE, INT_TYPE> & from, 
            host_mode_vector<FLOAT_TYPE, INT_TYPE> & to );


template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const host_mode_vector<FLOAT_TYPE, INT_TYPE> & from, 
            mode_vector<FLOAT_TYPE, INT_TYPE> & to );


// ===========================================================================



template<typename FLOAT_TYPE, typename INT_TYPE>
class mode_vector
{

  private:

  INT_TYPE noe;   ///> number of mesh elements
  INT_TYPE nompe; ///> max number of modes per mesh element for SEM = (N+1)
                  ///  per dimension

  FLOAT_TYPE * device_array;

  public:


  __device__ __host__ mode_vector () : noe(0),
                                       nompe(0),
                                       device_array(NULL)
  {}


 
  __host__ mode_vector (INT_TYPE _noe, INT_TYPE _nompe)
                       : noe(_noe), nompe(_nompe)
  {
    checkError (
      cudaMalloc (&device_array, noe*nompe*nompe*sizeof(FLOAT_TYPE)) );
  }



  __host__ mode_vector ( INT_TYPE _noe,
                         INT_TYPE _nompe,
                         FLOAT_TYPE * _d_a )
                       : noe(_noe),
                         nompe(_nompe),
                         device_array(_d_a)
  {}


  __host__ 
  mode_vector ( const host_mode_vector<FLOAT_TYPE, INT_TYPE> &  hmv )
              : device_array(NULL)
  {
    copy(hmv, *this);
  }


  // -----------------------------------------------------------------------

  
  __device__ inline FLOAT_TYPE & operator() (INT_TYPE i1,
                                             INT_TYPE i2,
                                             INT_TYPE e_idx)
  {

# if 0
    assert (device_array != NULL);
    assert (i1 < nompe && i2 < nompe);
    assert (e_idx < noe);
# endif

    return  device_array[ ( (i1*nompe+i2) * noe ) + e_idx ] ;
  }


  // -----------------------------------------------------------------------


  __host__ inline const FLOAT_TYPE * data() const
  {
    return device_array;
  }


  __host__ inline FLOAT_TYPE * data() 
  {
    return device_array;
  }


  __host__ inline INT_TYPE size() const
  {
    return noe*nompe*nompe;
  }


  // -----------------------------------------------------------------------


  __device__ __host__ inline INT_TYPE get_noe() const
  {
    return noe;
  }


  __device__ __host__ inline INT_TYPE get_nompe() const
  {
    return nompe;
  }


  // -----------------------------------------------------------------------


  __host__ inline INT_TYPE HOST_DEBUG (INT_TYPE i1,
                                       INT_TYPE i2,
                                       INT_TYPE e_idx)
  {
    return  ( ((i1*nompe)+i2) * noe ) + e_idx ;
  }


  // -----------------------------------------------------------------------


  __host__ inline void free()
  {
    checkError ( cudaFree( device_array ) );
    device_array = NULL;
    noe = 0;
    nompe = 0;
  }


  // -----------------------------------------------------------------------


  friend 
  void copy<FLOAT_TYPE, INT_TYPE>
           ( const host_mode_vector<FLOAT_TYPE, INT_TYPE> & from, 
             mode_vector<FLOAT_TYPE, INT_TYPE> & to );




};


#include<vector>

template<typename FLOAT_TYPE, typename INT_TYPE>
class host_mode_vector
{

  private:

  INT_TYPE noe;   ///> number of mesh elements
  INT_TYPE nompe; ///> max number of modes per mesh element for SEM = (N+1)
                  ///  per dimension

  std::vector<FLOAT_TYPE> host_vector;

  public:


  host_mode_vector () : noe(0),
                        nompe(0),
                        host_vector(0)
  {}

 
  host_mode_vector (INT_TYPE _noe, INT_TYPE _nompe)
                   : noe(_noe),
                     nompe(_nompe),
                     host_vector(_noe*_nompe*_nompe,0)
  {}

  host_mode_vector (const mode_vector<FLOAT_TYPE, INT_TYPE> & mv)
  {
    copy(mv, *this);
  }

  // -----------------------------------------------------------------------

  inline INT_TYPE get_noe() const
  {
    return noe;
  }


  inline INT_TYPE get_nompe() const
  {
    return nompe;
  }


  inline INT_TYPE size() const
  {
    return host_vector.size();
  }


  /** This function return a const pointer.
      In case a non const pointer is needed call directly
      host_vector.data() from a friend.  */

  inline const FLOAT_TYPE * data() const
  {
    return host_vector.data();
  } 

  // -----------------------------------------------------------------------
 
  inline FLOAT_TYPE & operator() (INT_TYPE i1,
                                  INT_TYPE i2,
                                  INT_TYPE e_idx)
  {
    assert (i1 < nompe && i2 < nompe);
    assert (e_idx < noe);

    return  host_vector [ ( (i1*nompe+i2) * noe ) + e_idx ] ;
  }

  // -----------------------------------------------------------------------

  friend 
  void copy<FLOAT_TYPE, INT_TYPE>
           ( const mode_vector<FLOAT_TYPE, INT_TYPE> & from, 
             host_mode_vector<FLOAT_TYPE, INT_TYPE> & to );


  friend
  void copy<FLOAT_TYPE, INT_TYPE>
           ( const host_mode_vector<FLOAT_TYPE, INT_TYPE> & from,
             mode_vector<FLOAT_TYPE, INT_TYPE> & to );

};





// ===========================================================================



template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const mode_vector<FLOAT_TYPE, INT_TYPE> & from, 
            host_mode_vector<FLOAT_TYPE, INT_TYPE> & to )
{

  const INT_TYPE size = from.size();
  to.noe = from.get_noe();
  to.nompe = from.get_nompe();

  to.host_vector.resize(size);

  // cudaMemcpy waits until the computation on the device is finished

  checkError ( cudaMemcpy( to.host_vector.data(), // to
                           from.data(),  // from
                           size*sizeof(FLOAT_TYPE),
                           cudaMemcpyDeviceToHost) );
    
}



template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const host_mode_vector<FLOAT_TYPE, INT_TYPE> & from, 
            mode_vector<FLOAT_TYPE, INT_TYPE> & to )
{

  to.noe = from.get_noe();
  to.nompe = from.get_nompe();

  const INT_TYPE size = from.get_noe()*from.get_nompe()*from.get_nompe(); 

  FLOAT_TYPE * old_device_array = to.device_array;

  checkError ( cudaMalloc (&to.device_array, size*sizeof(FLOAT_TYPE)) );

  // cudaMemcpy waits until the computation on the device is finished

  checkError ( cudaMemcpy( to.device_array,  // to
                           from.host_vector.data(),  // from
                           size*sizeof(FLOAT_TYPE),
                           cudaMemcpyHostToDevice) );

  checkError ( cudaFree(old_device_array) );
    
}



#endif
