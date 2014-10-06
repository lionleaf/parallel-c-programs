all : region

run : region
	qrsh -cwd mpirun -n 1 region pic1.bmp

region : region.c
	mpicc -std=c99 region.c bmp.c -o region -lm
	

