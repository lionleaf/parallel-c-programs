Parallel C programs
===================

Exercises from a course at the Norwegian University of Science and Technology in parallel computing.

1. Introduction
-------------------
Meant as a simple introduction to C to get us started, this exercise implements some matrix related calculations in C.

2. Region growing with MPI
-------------------
In this and many of the following exercises, we were asked to solve a region growing problem. The problem is as follows. You have a 2 dimensional grid of pixel and each pixel has a grayscale color (a byte between 0 and 255). Given a treshold and some positions (seeds), you are to "grow" the seeds to neighbouring pixels if the difference in color is below the threshold. Keep growing the region until it cannot grow any more.

In this exercise we were to implement it using **MPI**, a message passing protocol for parallel computations.

3. Serial optimization
-------------------

To quote the exercise: " In this problem, you should wirte a function that performs sparse matrix vector multiplication as fast as possible. You should do this by implementing the multiply function in the file spmv.c. Your function should be faster than the included multiply naive. You may create your own sparse matrix format,
using the struct s matrix t."

4. Multithreading with pthreads and OpenMP
-------------------
Two implementations of a parallel histogram equalization algorithm. One using **pthreads**, and the other using **OpenMP**.


5. Region growing in CUDA
-------------------
Same as in 2., only now we had 3 dimensions, and had to solve it with **CUDA**.

One of the few exercises I actually had the time to finish properly. ([The Computer Design Project](https://github.com/dmpro2014) often got in the way)

6. Region growing in OpenCL
-------------------
Same as 5, but using **OpenCL** instead of CUDA.

