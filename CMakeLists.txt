cmake_minimum_required(VERSION 2.8)

project(the_code)

find_package(CUDA REQUIRED)

add_executable(plot_basis validation_tests/sem_basis/plot_basis.cpp)

set( COMPUTE_CAPABILITY "sm_21" CACHE STRING "Set the device compute capability: sm_20, sm_35 ..." )
list( APPEND CUDA_NVCC_FLAGS -arch=${COMPUTE_CAPABILITY} )

include_directories("include/")



### PERFORMANCE tests

set( PERFORMANCE_TEST_EXECUTABLES 
      constant_dof_change_degree_and_noe
      flux_kernels_performance_test
      one_iteration_of_sem_sipg_solved_on_gpu_test
      volume_kernel_performance_test )

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
endforeach()

cuda_add_executable( multigpu_mesh_test validation_tests/mesh/multigpu_mesh_test.cu )
cuda_add_executable( rhs_test validation_tests/mesh/rhs_test.cu )



### VALIDATION tests LAPACK

## lapack wrapper

add_library(lapack_wrapper STATIC include/lapack_wrapper.cpp)
target_link_libraries(lapack_wrapper lapacke) # lapacke lib must be in the search path


add_executable(sipg_sem_1d_class_validation_test validation_tests/sipg_sem/sipg_sem_1d_class_validation_test.cpp)
target_link_libraries(sipg_sem_1d_class_validation_test lapack_wrapper)

add_executable(sem_1d_nitsche_bc_validation_test validation_tests/sem_basis/sem_1d_nitsche_bc_validation_test.cpp)
target_link_libraries(sem_1d_nitsche_bc_validation_test lapack_wrapper)

add_executable(sem_2d_nitsche_bc_validation_test validation_tests/sem_basis/sem_2d_nitsche_bc_validation_test.cpp)
target_link_libraries(sem_2d_nitsche_bc_validation_test lapack_wrapper)



### VALIDATION tests OTHERS





### DOC

find_package(Doxygen)
if(DOXYGEN_FOUND)
   add_custom_target( doc  
                        COMMAND make
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} )
endif()




#target_link_libraries
