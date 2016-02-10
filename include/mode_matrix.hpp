/*
This file provide classes to handle sparse stiffness matrices on host and
device. In this way only the values of matrix elements are stored, thera are
not indeces vector.

Elements are stored in this way.
    0        1                N
+-------+----------+-   -+----------+-   -+--
| i1,i2 | i1,ix/i2 | ... | ix/i1,i2 | ... |
+-------+----------+-   -+----------+-   -+--

*/


#ifndef __MODE_MATRIX_HPP__
#define __MODE_MATRIX_HPP__

#include<cassert>
#include<laplacian_operator.hpp>
#include<CUDA_ERRORS.hpp>


// ===========================================================================


template<typename FLOAT_TYPE, typename INT_TYPE>
class host_mode_matrix;

template<typename FLOAT_TYPE, typename INT_TYPE>
class mode_matrix;


template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const mode_matrix<FLOAT_TYPE, INT_TYPE> & from, 
            host_mode_matrix<FLOAT_TYPE, INT_TYPE> & to );


template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const host_mode_matrix<FLOAT_TYPE, INT_TYPE> & from, 
            mode_matrix<FLOAT_TYPE, INT_TYPE> & to );


// ===========================================================================


/**

  This class describes a data type which contains local operators for a given
  number of mesh elements. For now, the local operators have only the sparsity
  of Laplacian operator.  

*/
template<typename FLOAT_TYPE, typename INT_TYPE>
class mode_matrix
{

  private:

  INT_TYPE noe;   ///> number of mesh elements
  INT_TYPE nzpr;  ///> non zero elements per row
  INT_TYPE nompe; ///> max number of mode per element

  FLOAT_TYPE * device_array;

  public:


  __device__ __host__ mode_matrix() : noe(0),
                                      nompe(0),
                                      nzpr(0),
                                      device_array(NULL)
  {}


 
  /**

    This constructor allocate an empty data structure in device memory.
    CAUTION: The memory is allocated, but it is not initialized.
    At the beginning device_array contains garbage. Fill it before use.

    \param _noe number of mesh elements 
    \param _nompe number of element per mesh element
    \param _nzpr non-zero values per matrix row 

  */
  __host__ mode_matrix (INT_TYPE _noe, INT_TYPE _nompe, INT_TYPE _nzpr)
                       : noe(_noe), nompe(_nompe), nzpr(_nzpr)
  {
    checkError (
      cudaMalloc (&device_array, noe*nzpr*nompe*nompe*sizeof(FLOAT_TYPE)) );
  }


  /**

    This constructor map the data structure already present in device memory.

    \param _noe number of mesh elements 
    \param _nompe number of element per mesh element
    \param _nzpr non-zero values per matrix row 

    \param _d_a device memory pointer to data

  */
  __host__ mode_matrix ( INT_TYPE _noe,
                         INT_TYPE _nompe,
                         INT_TYPE _nzpr,
                         FLOAT_TYPE * _d_a )
                       : noe(_noe),
                         nompe(_nompe),
                         nzpr(_nzpr),
                         device_array(_d_a)
  {}

  /**

    This copy constructor copies to device memory an host_mode_matrix    

    \param hmm host_mode_matrix to copy on device

  */
  __host__ 
  mode_matrix ( const host_mode_matrix<FLOAT_TYPE, INT_TYPE> &  hmm )
              : device_array(NULL)
  {
    copy(hmm, *this);
  }


  // -----------------------------------------------------------------------

  /**
  This function return the pointer to the fist matrix value to the row
  */
  __device__ inline FLOAT_TYPE * operator() (INT_TYPE i1,
                                             INT_TYPE i2,
                                             INT_TYPE e_idx)
  {

# if 0
    assert (device_array != NULL);
    assert (i1 < nompe && i2 < nompe);
    assert (e_idx < noe);
# endif

    return  device_array + ( (i1*nompe+i2) * nzpr * noe  ) + e_idx ;
  }

  /**
  Sum .jump_to_next() to a pointer to obtain the pointer to the next
  element in the same row.
  */
  __device__ inline INT_TYPE jump_to_next() const
  {
    return noe;
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
    return noe*nompe*nompe*nzpr;
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


  __device__ __host__ inline INT_TYPE get_nzpr() const
  {
    return nzpr;
  }


  // -----------------------------------------------------------------------


  __host__ inline INT_TYPE HOST_DEBUG (INT_TYPE i1,
                                       INT_TYPE i2,
                                       INT_TYPE e_idx)
  {
    return  ( ((i1*nompe)+i2) * noe * nzpr ) + e_idx ;
  }


  // -----------------------------------------------------------------------


  __host__ inline void free()
  {
    checkError ( cudaFree( device_array ) );
    device_array = NULL;
    noe = 0;
    nompe = 0;
    nzpr = 0;
  }


  // -----------------------------------------------------------------------


  friend 
  void copy<FLOAT_TYPE, INT_TYPE>
           ( const host_mode_matrix<FLOAT_TYPE, INT_TYPE> & from, 
             mode_matrix<FLOAT_TYPE, INT_TYPE> & to );




};


#include<vector>
/**

  The host_mode_matrix can handle local matrices.
  An object of this type can handle local operator for a given
  number of elements. 

*/
template<typename FLOAT_TYPE, typename INT_TYPE>
class host_mode_matrix
{

  private:

  INT_TYPE noe;   ///>  number of mesh elements
  INT_TYPE nompe; ///>  number of non-zero per local matrix
  INT_TYPE nzpr;  ///>  non zero elements per row

  std::vector<FLOAT_TYPE> host_matrix;
 
  // -----------------------------------------------------------------------


  public:


  host_mode_matrix () : noe(0),
                        nompe(0),
                        nzpr(0),
                        host_matrix(0)
  {}

 
  host_mode_matrix (INT_TYPE _noe, INT_TYPE _order)
                   : noe(_noe),
                     nompe(_order+1)
  {

    nzpr = 2*_order + 1;
    host_matrix.resize(noe*nompe*nompe*nzpr);

  }

  host_mode_matrix (const mode_matrix<FLOAT_TYPE, INT_TYPE> & mv)
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


  inline INT_TYPE get_nzpr() const
  {
    return nzpr;
  }


  inline INT_TYPE size() const
  {
    return host_matrix.size();
  }


  /** This function return a const pointer.
      In case a non const pointer is needed call directly
      host_matrix.data() from a friend.  */

  inline const FLOAT_TYPE * data() const
  {
    return host_matrix.data();
  } 

  // -----------------------------------------------------------------------

  inline FLOAT_TYPE & operator() (INT_TYPE i1,
                                  INT_TYPE i2,
                                  INT_TYPE r_idx,
                                  INT_TYPE e_idx)
  {
    assert (i1 < nompe && i2 < nompe);
    assert (e_idx < noe);

    return  host_matrix [ ( (i1*nompe+i2) * nzpr + r_idx ) * noe + e_idx ] ;
  }

  // -----------------------------------------------------------------------

  friend 
  void copy<FLOAT_TYPE, INT_TYPE>
           ( const mode_matrix<FLOAT_TYPE, INT_TYPE> & from, 
             host_mode_matrix<FLOAT_TYPE, INT_TYPE> & to );


  friend
  void copy<FLOAT_TYPE, INT_TYPE>
           ( const host_mode_matrix<FLOAT_TYPE, INT_TYPE> & from,
             mode_matrix<FLOAT_TYPE, INT_TYPE> & to );

};





// ===========================================================================


/**
  This function copies a `mode_matrix` from the device to the host. 
*/
template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const mode_matrix<FLOAT_TYPE, INT_TYPE> & from, 
            host_mode_matrix<FLOAT_TYPE, INT_TYPE> & to )
{

  const INT_TYPE size = from.size();
  to.noe = from.get_noe();
  to.nompe = from.get_nompe();
  to.nzpr = from.get_nzpr();

  to.host_matrix.resize(size);

  // cudaMemcpy waits until the computation on the device is finished

  checkError ( cudaMemcpy( to.host_matrix.data(), // to
                           from.data(),  // from
                           size*sizeof(FLOAT_TYPE),
                           cudaMemcpyDeviceToHost) );
    
}


/**
  This function copies a `mode_matrix` from the host to the device. 
*/
template<typename FLOAT_TYPE, typename INT_TYPE>
void copy ( const host_mode_matrix<FLOAT_TYPE, INT_TYPE> & from, 
            mode_matrix<FLOAT_TYPE, INT_TYPE> & to )
{

  to.noe = from.get_noe();
  to.nompe = from.get_nompe();
  to.nzpr = from.get_nzpr();

  const INT_TYPE size = from.size();

  FLOAT_TYPE * old_device_array = to.device_array;

  checkError ( cudaMalloc (&to.device_array, size*sizeof(FLOAT_TYPE)) );

  // cudaMemcpy waits until the computation on the device is finished

  checkError ( cudaMemcpy( to.device_array,  // to
                           from.host_matrix.data(),  // from
                           size*sizeof(FLOAT_TYPE),
                           cudaMemcpyHostToDevice) );

  checkError ( cudaFree(old_device_array) );
    
}



// ===========================================================================

/**
  This class creates a host_mode_matrix containing the local Laplacian operators. 
*/
template<typename FLOAT_TYPE, typename INT_TYPE>
class host_laplacian_matrix : public  host_mode_matrix<FLOAT_TYPE, INT_TYPE>
{
  
  public:
/**
  \param _noe number of elements
  \param _order degree of basis functions 
*/
  host_laplacian_matrix (INT_TYPE _noe, INT_TYPE _order)
   : host_mode_matrix<FLOAT_TYPE, INT_TYPE>( _noe, _order)
  {

    laplace_2d<FLOAT_TYPE> lap(_order);

    for(int el = 0; el < _noe; el++)
      for(int i1 = 0; i1 <= _order; ++i1)
        for(int i2 = 0; i2 <= _order; ++i2)
        {

         // i1,i2
         (*this)(i1,i2,0,el) = lap(i1,i2,i1,i2);


         int r = 0;

         // i1,[0..N]/i2
         for( int j = 0; j <= _order; ++j )
           if (j != i2)
           {
             r++;
             (*this)(i1,i2,r,el) =  lap(i1,i2,i1,j);
           }

         // [0..N]/i1,i2
         for( int j = 0; j <= _order; ++j )
           if (j != i1)
           {
             r++;
             (*this)(i1,i2,r,el) =  lap(i1,i2,j,i2);
           }

        }

  }


};


#endif
