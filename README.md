CUDA SIPG matrix-free
=====================

This code solves the Poisson problem with Dirichlet border condition with the
Symmetric Interior Penalty Galerkin method.

The code has been written in CUDA C++.

Requisites
----------

 * a CUDA device with compute capability 2.0 or higher.
 * CUDA Toolkit 5.0 or higher.
 * cmake 2.8 or higher.
 * lapacke library is needed to run some preliminary validation tests on CPU, it is not necessary for the solution of the problem. The command `sudo apt-get install liblapacke-dev` installs it on Debian or Ubuntu.
 * doxygen for build the documentation
 * GCL library is needed in order to run the code on multiple GPUs in a MPI environment. GCL requires Boost library. 

  
Build 
-----

```
    $ cd source-dir
    $ mkdir build
    $ cd build 
    $ cmake -D COMPUTE_CAPABILITY=sm_xx ..
```

Where `sm_xx` is the compute capability of your device, e.g. `sm_20`, `sm_35`...
In order to employ GCL library, you need to set `GCL_USAGE=ON` and `GCL_SOURCE_DIR`.


Now, you can compile and run the binary with


```
    $ make binary-name
    $ ./binary-name
```

The command `make help` shows the full list of available binaries.

A `Makefile` in the source root directory builds the documentation with `doxygen`  

```
    $ cd source-dir
    $ mkdir doc
    $ make
```

Now, you can find the generated documentation in `doc` directory.



Binary List
-----------

The most important binaries are

 * `sipg_2d_h_adaptivity_multigpu` is the two-dimensional Poisson problem validation tests. It uses multiple GPUs.
 * `sipg_2d_h_adaptivity` and `sipg_2d_p_adaptivity` are the two-dimensional Poisson problem validation tests. They use the GPU.
 * `sipg_sem_1d_class_validation_test` is a one-dimensional Poisson problem preliminary test. It uses the CPU and lapacke.
 * `sem_1d_nitsche_bc_validation_test` and `sem_2d_nitsche_bc_validation_test` are one and two-dimensional SEM basis validation tests. They use the CPU and lapacke. 
 * `constant_dof_change_degree_and_noe`, `flux_kernels_performance_test`, `one_iteration_of_sem_sipg_solved_on_gpu_test`, `volume_kernel_performance_test` are run with `nvprof` to test the performance. 



