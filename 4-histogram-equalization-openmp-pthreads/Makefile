all : histogram_serial histogram_omp histogram_pthreads

histogram_serial : histogram_serial.c
	gcc -std=c99 -Wall histogram_serial.c bmp.c -o histogram_serial
    
histogram_omp : histogram_omp.c
	gcc -std=c99 -Wall histogram_omp.c bmp.c -o histogram_omp -fopenmp
    
histogram_pthreads : histogram_pthreads.c
	gcc -std=c99 -Wall histogram_pthreads.c bmp.c -o histogram_pthreads -pthread
	
clean :
	rm histogram_omp histogram_serial histogram_pthreads
