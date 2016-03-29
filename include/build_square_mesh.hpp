#ifndef __SQUARE_MESH_HPP__IWFHNQAZH
#define __SQUARE_MESH_HPP__IWFHNQAZH


#include<vector>
#include<mesh_info.hpp>

#include<cassert>


/*

 dim = 3
 noe = 9

(0,1)             (1,1)
   _____ _____ _____
  |     |     |     |
  |  0  |  1  |  2  |
  |_____|_____|_____|
  |     |     |     |
  |  3  |  4  |  5  |
  |_____|_____|_____|
  |     |     |     |
  |  6  |  7  |  8  |
  |_____|_____|_____|

(0,0)              (1,0)

*/

#define __QUADRILATERAL_MESH_INFO__

/**
 This class builds (on host) the data sets which describes
 a two-dimensional Cartesian mesh on (0,1)X(0,1).
 `FLOAT_TYPE` is the floating point data type employed.
*/
template<typename FLOAT_TYPE>
class square_mesh
{
  private:
    int _dim; //< number of elements along the edge
    int _noe; //< the total number of mesh element _noe = _dim * _dim

#ifndef  __QUADRILATERAL_MESH_INFO__
    std::vector<int> _neighbourhood;
    void compute_neighbours();
#endif

    std::vector<FLOAT_TYPE> _x_coord; //< vector of x coordinates of mesh elements
    std::vector<FLOAT_TYPE> _y_coord; //< vector of y coordinates of mesh elements

    void compute_coordinates();


  public:

#ifdef  __QUADRILATERAL_MESH_INFO__
    quadrilateral_mesh_info<FLOAT_TYPE> device_info; //< mesh information on device
#else
    mesh_info<FLOAT_TYPE> device_info;
#endif

    square_mesh() {}

/**
  
  \param dim  number of elements per edge
*/
    square_mesh(int dim)
       : _dim(dim),
         _noe(dim*dim),
#ifndef  __QUADRILATERAL_MESH_INFO__
         _neighbourhood(4*_noe),
#endif
         _x_coord(4*_noe),
         _y_coord(4*_noe)
    {

      compute_coordinates();

#ifdef  __QUADRILATERAL_MESH_INFO__

      quadrilateral_mesh_info<FLOAT_TYPE> m( _dim, _dim,
                                             _x_coord.data(),
                                             _y_coord.data() );
#else
      compute_neighbours();
      mesh_info<FLOAT_TYPE> m( _noe,
                               _x_coord.data(),
                               _y_coord.data(),
                               _neighbourhood.data() );
#endif

      device_info = m;

    }

    int noe() const
    {
      return _noe;
    }

    int dim() const
    {
      return _dim;
    }

    const std::vector<FLOAT_TYPE> & x_coord() const
    {
      return _x_coord;
    }

    const std::vector<FLOAT_TYPE> & y_coord() const
    {
      return _y_coord;
    }


    // ------------------------


   
    const FLOAT_TYPE &
    get_x (int c, int element) const
    {

      // c=0 -> x1
      // c=1 -> x2
      // c=2 -> x3
      // c=3 -> x4

      return _x_coord.data()[c*_noe + element];

    }


    const FLOAT_TYPE &
    get_y (int c, int element) const
    {

      // c=0 -> y1
      // c=1 -> y2
      // c=2 -> y3
      // c=3 -> y4

      return _y_coord.data()[c*_noe + element];

     }


    // ---------------------------------------------------------
 
#ifndef  __QUADRILATERAL_MESH_INFO__

    const std::vector<int> & neighbourhood() const
    {
      return _neighbourhood;
    }

#endif
 
    int
    get_neighbor (int n, int element) const
    {

#ifdef  __QUADRILATERAL_MESH_INFO__

      const int x = element % _dim;
      const int y = element / _dim;

      int retval;

      switch(n)
      { 
        case 0:
          retval = device_info.get_neighborhood_DOWN(x, y); 
        case 1:
          retval = device_info.get_neighborhood_RIGHT(x, y);
        case 2:
          retval = device_info.get_neighborhood_UP(x, y); 
        case 3: 
          retval = device_info.get_neighborhood_LEFT(x, y);
        default:
          assert(true);
      }

      return retval;

#else 
      // DOWN   n = 0
      // RIGHT  n = 1
      // UP     n = 2
      // LEFT   n = 3

      return _neighbourhood.data()[n*_noe + element];
#endif
    }


};



#ifndef  __QUADRILATERAL_MESH_INFO__

template<typename FLOAT_TYPE>
void square_mesh<FLOAT_TYPE>::compute_neighbours()
{

  // DOWN RIGHT UP LEFT

  // DOWN 
  for (int i = 0; i < _dim; ++i)
    for (int j = 0; j < _dim; ++j)
      if ( i == _dim-1 )
        _neighbourhood[_dim*i + j] = -1;
        else _neighbourhood[_dim*i + j] = (_dim*i + j) + _dim;

  // RIGHT
  for (int i = 0; i < _dim; ++i)
    for (int j = 0; j < _dim; ++j)
      if ( j == _dim-1 )
        _neighbourhood[_noe + _dim*i + j] = -2;
        else _neighbourhood[_noe + _dim*i + j] = (_dim*i + j) + 1;

  // UP
  for (int i = 0; i < _dim; ++i)
    for (int j = 0; j < _dim; ++j)
      if ( i == 0 )
        _neighbourhood[2*_noe + _dim*i + j] = -3;
        else _neighbourhood[2*_noe + _dim*i + j] = (_dim*i + j) - _dim;

  // LEFT
  for (int i = 0; i < _dim; ++i)
    for (int j = 0; j < _dim; ++j)
      if ( j == 0 )
        _neighbourhood[3*_noe + _dim*i + j] = -4;
        else _neighbourhood[3*_noe + _dim*i + j] = (_dim*i + j) - 1;


}

#endif

/**
  This method computes the coordinates of mesh elements.
*/
template<typename FLOAT_TYPE>
void square_mesh<FLOAT_TYPE>::compute_coordinates()
{

  FLOAT_TYPE delta = 1./FLOAT_TYPE(_dim);
  
  FLOAT_TYPE xx(0), yy(1);


  for (int i = 0; i < _dim; ++i )
  {
    xx = 0;
    yy -= delta;

    for (int j = 0; j < _dim; ++j )
    {

      int id = i*_dim + j;

      _x_coord[0*_noe + id] = xx;
      _y_coord[0*_noe + id] = yy;

      _x_coord[1*_noe + id] = xx + delta;
      _y_coord[1*_noe + id] = yy;

      _x_coord[2*_noe + id] = xx + delta; 
      _y_coord[2*_noe + id] = yy + delta;

      _x_coord[3*_noe + id] = xx;
      _y_coord[3*_noe + id] = yy + delta; 


      xx += delta;

    }

  }

}








// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



template<typename FLOAT_TYPE>
class square_mesh_multigpu
{

  private:

    int _local_dim; //< number of element per node _with_ _halos_
    int _total_dim; //< total number of elements on the edge of the entire (all node) mesh
    int _grid_size; //< number of tails per edge 
    int _grid_coord_x; // need to be < _grid_size 
    int _grid_coord_y; // need to be < _grid_size 

    bool _DOWNborder;
    bool _RIGHTborder;
    bool _UPborder;
    bool _LEFTborder;

    std::vector<FLOAT_TYPE> _x_coord;
    std::vector<FLOAT_TYPE> _y_coord;

    void compute_coordinates();
    void compute_if_local_mesh_touch_global_domain_edges();

  public:

    local_quadrilateral_mesh_info<FLOAT_TYPE> device_info;

    square_mesh_multigpu() {}

    square_mesh_multigpu( int local_dim_without_halos,
                          int grid_size,
                          int grid_coord_x, 
                          int grid_coord_y )  
                        : _local_dim(local_dim_without_halos + 2),
                          _grid_size(grid_size),
                          _grid_coord_x(grid_coord_x),
                          _grid_coord_y(grid_coord_y),
                          _x_coord(4*_local_dim*_local_dim),
                          _y_coord(4*_local_dim*_local_dim),
                          _total_dim(grid_size*local_dim_without_halos)
    {

      assert( _grid_coord_x < _grid_size && _grid_coord_y < _grid_size );

      compute_if_local_mesh_touch_global_domain_edges();
      compute_coordinates();


      local_quadrilateral_mesh_info<FLOAT_TYPE> m( _local_dim, 
                                                   _local_dim,
                                                   _x_coord.data(),
                                                   _y_coord.data(),
                                                   _DOWNborder,
                                                   _RIGHTborder,
                                                   _UPborder,
                                                   _LEFTborder );

      device_info = m;

    }


    inline int dim() const
    {
      return _total_dim;
    }

    inline int local_dim() const
    {
      return _local_dim;
    }

    /**
     number of elements of the tile
    */
    inline int noe() const
    {
      return _local_dim*_local_dim;
    }


    /** 
      \return number of tails per edge
    */
    inline int grid_size() const
    {
      return _grid_size; 
    }
 
    inline int grid_coord_x() const
    {
      return _grid_coord_x;
    }
 
    inline int grid_coord_y() const
    {
      return _grid_coord_y;
    }



    FLOAT_TYPE 
    get_x (int c, int element) const
    {

      // c=0 -> x1
      // c=1 -> x2
      // c=2 -> x3
      // c=3 -> x4

      return _x_coord[element];

    }


    FLOAT_TYPE
    get_y (int c, int element) const
    {

      // c=0 -> y1
      // c=1 -> y2
      // c=2 -> y3
      // c=3 -> y4

      return _y_coord[element];

     }


    const std::vector<FLOAT_TYPE> & x_coord() const
    {
      return _x_coord;
    }

    const std::vector<FLOAT_TYPE> & y_coord() const
    {
      return _y_coord;
    }

   int
    get_neighbor (int n, int element) const
    {

      const int x = element % _local_dim;
      const int y = element / _local_dim;

      int retval;
   
      switch(n)
      { 
        case 0:
         retval = device_info.get_neighborhood_DOWN(x, y); 
        case 1:
         retval = device_info.get_neighborhood_RIGHT(x, y);
        case 2:
         retval = device_info.get_neighborhood_UP(x, y); 
        case 3: 
         retval =  device_info.get_neighborhood_LEFT(x, y);
        default:
          assert(true);
      }

      return retval;

   }

    bool DOWNborder()  const { return _DOWNborder; }
    bool RIGHTborder() const { return _RIGHTborder; }
    bool UPborder()    const { return _UPborder; }
    bool LEFTborder()  const { return _LEFTborder; }


};


template<typename FLOAT_TYPE>
void square_mesh_multigpu<FLOAT_TYPE>::compute_coordinates()
{

  FLOAT_TYPE delta = 1./FLOAT_TYPE(_total_dim);
  FLOAT_TYPE DELTA = 1./FLOAT_TYPE(_grid_size);

  // set the point on local domain

  FLOAT_TYPE yy( 1 - _grid_coord_y*DELTA );

  for (int i = 0; i < _local_dim; ++i )
  {

    FLOAT_TYPE xx( _grid_coord_x*DELTA - delta );

    for (int j = 0; j < _local_dim; ++j )
    {

      int id = i*_local_dim + j;

      _x_coord[id] = xx;
      _y_coord[id] = yy;

      xx += delta;

    }

    yy -= delta;

  }

}


template<typename FLOAT_TYPE>
void square_mesh_multigpu<FLOAT_TYPE>::compute_if_local_mesh_touch_global_domain_edges()
{
  
  if (_grid_coord_y == 0) _UPborder = true;
    else _UPborder = false;
 
  if (_grid_coord_y == _grid_size-1 ) _DOWNborder = true;
    else _DOWNborder = false;
 
  if (_grid_coord_x == _grid_size-1) _RIGHTborder = true;
    else _RIGHTborder = false;

  if (_grid_coord_x ==  0) _LEFTborder = true;
    else _LEFTborder = false;

  return;
}


#endif
