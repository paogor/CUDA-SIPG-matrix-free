#ifndef __MESH_INFO_HPP__
#define __MESH_INFO_HPP__

#include<CUDA_ERRORS.hpp>

/*

REFERENCE element
  (-1,1)------(1,1)
     |##########|        
     |##########|     
     |##########| 
  (-1,-1)-----(1,-1)


ACTUAL element
     (x4,y4)----(x3,y3)
       /##########/        
      /## area ##/     
     /##########/ 
  (x1,y1)----(x2,y2)

 
Jk:    Jacobian matrix - REFERENCE to ACTUAL
Jk^-1: Jacobian matrix - ACTUAL to REFERENCE

If ACTUAL elements have no curved edges,
Jacobian matrices have constant entries.



mesh_info:

+++ double/float +++

area
x1 ,y1
x2 ,y2
x3 ,y3
x4 ,y4

det of JACOBIAN matrix



+++ int +++ 

basis order in each direction

neighbor DOWN   1
neighbor RIGHT  2
neighbor UP     3
neighbor LEFT   4

*/

/** 
 A generic mesh information structure (on **device**) for quadrilateral
 elements.
 For now deprecated, use `quadrilateral_mesh_info` instead.
*/
template<typename FLOAT_TYPE>
class mesh_info
{

  private: 

  int noe; ///> number of elements in the mesh

  FLOAT_TYPE * x; ///> poiter to an array of size (num_of_elem * 4)
                  /// x[c*noe + e] is the x_c coordinate of element number e 
  FLOAT_TYPE * y; ///> same as above

  int * neighborhood; 

  public:

  __host__ mesh_info ( int _noe,
                       FLOAT_TYPE * h_x,
                       FLOAT_TYPE * h_y, 
                       int * h_neighborhood )
                     : noe(_noe) 
  {


    const size_t byte_size = 4*noe*sizeof(FLOAT_TYPE);

    checkError (cudaMalloc (&x, byte_size));
    checkError (cudaMalloc (&y, byte_size));
    checkError (cudaMalloc (&neighborhood, 4*noe*sizeof(int)));

    checkError (cudaMemcpy (x, h_x, byte_size, cudaMemcpyHostToDevice));
    checkError (cudaMemcpy (y, h_y, byte_size, cudaMemcpyHostToDevice));
    checkError (cudaMemcpy (neighborhood, h_neighborhood, 4*noe*sizeof(int), cudaMemcpyHostToDevice));

  }


  __host__ __device__ mesh_info(){} 


  // ---------------------------------------------------------

  __host__ __device__ inline int 
  get_noe () const
  {
    return noe; 
  }

  // ---------------------------------------------------------

  __device__ inline const FLOAT_TYPE &
  get_x (int c, int element) const
  {

    // c=0 -> x1
    // c=1 -> x2
    // c=2 -> x3
    // c=3 -> x4

    return x [c*noe + element];
  }

  __device__ inline const FLOAT_TYPE &
  get_y (int c, int element) const
  {

    // c=0 -> y1
    // c=1 -> y2
    // c=2 -> y3
    // c=3 -> y4

    return y [c*noe + element];
  }


  // ---------------------------------------------------------
  
  __device__ inline const int &
  get_neighborhood (int n, int element) const
  {

    // DOWN   n = 1
    // RIGHT  n = 2
    // UP     n = 3
    // LEFT   n = 4

    return neighborhood [n*noe + element];
  }

  // ---------------------------------------------------------

  /** It frees the device memory */
  __host__ inline void free()
  {
    // free device memory and set to NULL pointers

    cudaFree(x); 
    cudaFree(y);
    cudaFree(neighborhood); 

  }

};

/**
 A generic mesh information structure (on **device**) for quadrilateral
 elements.
 `FLOAT_TYPE` is the float point data type of element coordinates.
*/
template<typename FLOAT_TYPE>
class quadrilateral_mesh_info
{

  private: 

  int dimx; ///> number of elements on the x coord
  int dimy; ///> number of elements on the y coord

  int noe; ///> total number of elements in the mesh noe = dimx*dimy

  FLOAT_TYPE * x; ///> poiter to an array of size (num_of_elem * 4)
                  /// x[c*noe + e] is the x_c coordinate of element number e 
  FLOAT_TYPE * y; ///> same as above

  public:

/**

  \param _dimy number of elements on the y coord
  \param _dimx number of elements on the x coord
  \param h_x hosts pointer to x coords of the mesh elements
  \param h_y hosts pointer to y coords of the mesh elements
*/
  __host__ quadrilateral_mesh_info ( int _dimx, int _dimy,
                                     FLOAT_TYPE * h_x,
                                     FLOAT_TYPE * h_y )
                                   : noe(_dimx*_dimy), dimx(_dimx), dimy(_dimy)  
  {


    const size_t byte_size = 4*noe*sizeof(FLOAT_TYPE);

    checkError (cudaMalloc (&x, byte_size));
    checkError (cudaMalloc (&y, byte_size));

    checkError (cudaMemcpy (x, h_x, byte_size, cudaMemcpyHostToDevice));
    checkError (cudaMemcpy (y, h_y, byte_size, cudaMemcpyHostToDevice));

  }


  __host__ __device__ quadrilateral_mesh_info(){} 


  __host__ __device__ inline int
  get_dimx() const
  {
    return dimx;
  }


  __host__ __device__ inline int
  get_dimy() const
  {
    return dimy;
  }



  __host__ __device__ inline int
  compute_idx(int x, int y)
  {
    return dimx*y + x; 
  } 


  // ---------------------------------------------------------

  __host__ __device__ inline int 
  get_noe () const
  {
    return noe; 
  }

  // ---------------------------------------------------------

  __device__ inline const FLOAT_TYPE &
  get_x (int c, int element) const
  {

    // c=0 -> x1
    // c=1 -> x2
    // c=2 -> x3
    // c=3 -> x4

    return x [c*noe + element];
  }

  __device__ inline const FLOAT_TYPE &
  get_y (int c, int element) const
  {

    // c=0 -> y1
    // c=1 -> y2
    // c=2 -> y3
    // c=3 -> y4

    return y [c*noe + element];
  }


  // ---------------------------------------------------------

  /** 
   \return index of the neighbor at the bottom
  */ 
  __host__ __device__ inline int
  get_neighborhood_DOWN (int x, int y) const
  {
    return ( dimy-1 == y ? -1 : dimx*(y+1) + x );
  }

  /** 
   \return index of the neighbor at the right
  */  
  __host__ __device__ inline int 
  get_neighborhood_RIGHT (int x, int y) const
  {
    return ( dimx-1 == x ? -2 : dimx*y + x + 1 ); 
  }

  /** 
   \return index of the neighbor at the top
  */ 
  __host__ __device__ inline int 
  get_neighborhood_UP (int x, int y) const
  {
    return ( 0 == y ? -3 : dimx*(y-1) + x );
  }
  
  /** 
   \return index of the neighbor at the left
  */ 
  __host__ __device__ inline int
  get_neighborhood_LEFT (int x, int y) const
  {
    return ( 0 == x ? -4 : dimx*y + x - 1 );
  }


  // ---------------------------------------------------------

  /** It frees the device memory */
  __host__ inline void free()
  {
    // free device memory and set to NULL pointers

    cudaFree(x); 
    cudaFree(y);

  }

};


// non-tested code follows 
/**
 A generic mesh information structure (on **device**) for quadrilateral
 elements for **multigpu**.
 `FLOAT_TYPE` is the float point data type of element coordinates.
*/
template<typename FLOAT_TYPE>
class local_quadrilateral_mesh_info : public quadrilateral_mesh_info<FLOAT_TYPE> 
{

  private:

    bool DOWNborder;
    bool RIGHTborder;
    bool UPborder;
    bool LEFTborder;

  public:

  __host__ local_quadrilateral_mesh_info ( int _dimy, int _dimx, /*_dimx and _dimy include halos*/
                                           FLOAT_TYPE * h_x, FLOAT_TYPE * h_y,
                                           bool _DOWNborder,
                                           bool _RIGHTborder,
                                           bool _UPborder,
                                           bool _LEFTborder )
  : quadrilateral_mesh_info<FLOAT_TYPE>::quadrilateral_mesh_info( _dimy, _dimx, h_x, h_y ),
    DOWNborder(_DOWNborder), 
    RIGHTborder(_RIGHTborder),
    UPborder(_UPborder),
    LEFTborder(_LEFTborder)
  {}

  __host__ __device__ local_quadrilateral_mesh_info(){} 


  __host__ __device__ inline int
  get_dimx() const // doesn't include halos
  {
    return quadrilateral_mesh_info<FLOAT_TYPE>::get_dimx() - 2;
  }


  __host__ __device__ inline int
  get_dimy() const // doesn't include halos
  {
    return quadrilateral_mesh_info<FLOAT_TYPE>::get_dimy() - 2;
  }



  __host__ __device__ inline int
  compute_idx(int x, int y) // x,y excludes halos, return the index considering the halos  
  {
    return quadrilateral_mesh_info<FLOAT_TYPE>::get_dimx()*(y+1) + (x+1); 
  } 

  // ---------------------------------------------------------

  
  __host__ __device__ inline int
  get_neighborhood_DOWN (int x, int y) const
  {
    return (DOWNborder && y == get_dimy()-1) ? -1 : quadrilateral_mesh_info<FLOAT_TYPE>::get_neighborhood_DOWN(x+1, y+1);
  }
  
  __host__ __device__ inline int 
  get_neighborhood_RIGHT (int x, int y) const
  {
    return (RIGHTborder && x == get_dimx()-1) ? -2 : quadrilateral_mesh_info<FLOAT_TYPE>::get_neighborhood_RIGHT(x+1, y+1);
  }
  
  __host__ __device__ inline int 
  get_neighborhood_UP (int x, int y) const
  {
    return (UPborder && y ==0) ? -3 : quadrilateral_mesh_info<FLOAT_TYPE>::get_neighborhood_UP(x+1, y+1);
  }
  
  __host__ __device__ inline int
  get_neighborhood_LEFT (int x, int y) const
  {
    return (LEFTborder && x==0) ? -4 : quadrilateral_mesh_info<FLOAT_TYPE>::get_neighborhood_LEFT(x+1, y+1);
  }


  // ---------------------------------------------------------


};


#endif
