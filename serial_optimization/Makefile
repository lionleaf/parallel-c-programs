run : spmv
	qrsh -cwd ./spmv 100000 401 200 100 200 10

spmv : spmv.c
	gcc -Wall -std=c99 -O3 -msse3 spmv.c -o spmv -lm
	
clean :
	rm spmv
