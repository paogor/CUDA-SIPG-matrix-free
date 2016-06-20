cmake_minimum_required(VERSION 2.8)

project(CUDA-Matrix-Free-SIPG)

find_package(CUDA REQUIRED)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set( COMPUTE_CAPABILITY "sm_21" CACHE STRING "Set the device compute capability: sm_20, sm_35 ..." )
list( APPEND CUDA_NVCC_FLAGS -arch=${COMPUTE_CAPABILITY} )

set( GCL_USAGE "OFF" CACHE BOOL "Set ON in order to employ GCL library" )
set( GCL_SOURCE_DIR "../gcl/trunk" CACHE STRING "Set the GCL source directory" )
set( DAINT_LAPACKE "OFF" CACHE BOOL "Set ON in order to employ LAPACKE library on DAINT")


if( "${DAINT_LAPACKE}" STREQUAL "ON" )
 include_directories("/users/paogor/lapack/lapack-3.6.0/LAPACKE/include")
 link_directories("/users/paogor/lapack/lapack-3.6.0/build/lib")
endif()



include_directories("include/")


## lapack wrapper
add_library(lapack_wrapper STATIC include/lapack_wrapper.cpp)
target_link_libraries(lapack_wrapper lapacke) # lapacke lib must be in the search path



### PERFORMANCE tests

set( PERFORMANCE_TEST_EXECUTABLES 
      constant_dof_change_degree_and_noe
      flux_kernels_performance_test
      one_iteration_of_sem_sipg_solved_on_gpu_test
      volume_kernel_performance_test 
      mvm_performance_test )

foreach( EXEC ${PERFORMANCE_TEST_EXECUTABLES} )
  cuda_add_executable( ${EXEC} performance_tests/${EXEC}.cu)
endforeach()



### VALIDATION tests CUDA

set( SIPG_VALIDATION_EXECUTABLES
      sipg_2d_h_adaptivity
      sipg_2d_p_adaptivity )

foreach( EXEC ${SIPG_VALIDATION_EXECUTABLES} )
  cuda_add_executable( ${EXEC} validation_tests/sipg_sem/${EXEC}.cu)
  cuda_add_cublas_to_target( ${EXEC} )
  target_link_libraries(${EXEC} lapack_wrapper)
endforeach()

cuda_add_executable( multigpu_mesh_test validation_tests/mesh/multigpu_mesh_test.cu )
cuda_add_executable( rhs_test validation_tests/mesh/rhs_test.cu )
cuda_add_executable( dotproduct_multigpu_test validation_tests/mesh/dotproduct_multigpu_test.cu )


## VALIDATION tests LAPACK


add_executable(sipg_sem_1d_class_validation_test validation_tests/sipg_sem/sipg_sem_1d_class_validation_test.cpp)
target_link_libraries(sipg_sem_1d_class_validation_test lapack_wrapper)

add_executable(sem_1d_nitsche_bc_validation_test validation_tests/sem_basis/sem_1d_nitsche_bc_validation_test.cpp)
target_link_libraries(sem_1d_nitsche_bc_validation_test lapack_wrapper)

add_executable(sem_2d_nitsche_bc_validation_test validation_tests/sem_basis/sem_2d_nitsche_bc_validation_test.cpp)
target_link_libraries(sem_2d_nitsche_bc_validation_test lapack_wrapper)

add_executable(diagonal_block_inversion validation_tests/sem_basis/diagonal_block_inversion.cpp)
target_link_libraries(diagonal_block_inversion lapack_wrapper)


### VALIDATION tests GCL 


if( "${GCL_USAGE}" STREQUAL "ON" )

  find_package( Boost )
    if( Boost_FOUND )
    include_directories( ${Boost_INCLUDE_DIR} )
  endif()

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GCL_GPU_ -D_GCL_MPI_ -DNDEBUG")

  include_directories( "${GCL_SOURCE_DIR}/L3/include/" ) ## Library header files
  include_directories( "${GCL_SOURCE_DIR}/L3/src/" )     ## Library source files
  include_directories( "${GCL_SOURCE_DIR}/L2/include/" ) ## Library header files
  include_directories( "${GCL_SOURCE_DIR}/L2/src/" )     ## Library source files
  include_directories( "${GCL_SOURCE_DIR}" )             ## Library header files

  ADD_LIBRARY( gcl STATIC "${GCL_SOURCE_DIR}/L3/include/GCL.h"
                          "${GCL_SOURCE_DIR}/L2/include/stats_collector.h"
                          "${GCL_SOURCE_DIR}/L3/src/GCL.cpp" )


  cuda_add_executable( sipg_2d_h_adaptivity_multigpu validation_tests/sipg_sem/sipg_2d_h_adaptivity_multigpu.cu )
  add_dependencies( sipg_2d_h_adaptivity_multigpu gcl ) 
  cuda_add_cublas_to_target( sipg_2d_h_adaptivity_multigpu )
  target_link_libraries( sipg_2d_h_adaptivity_multigpu gcl ${CUDA_MPI_LIB} )


  cuda_add_executable( multigpu-mvm performance_tests/multigpu-mvm.cu )
  add_dependencies( multigpu-mvm gcl ) 
  cuda_add_cublas_to_target( multigpu-mvm )
  target_link_libraries( multigpu-mvm gcl ${CUDA_MPI_LIB} )


endif()


### DOC

find_package(Doxygen)
if(DOXYGEN_FOUND)
   add_custom_target( doc  
                        COMMAND make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
endif()




#target_link_libraries
