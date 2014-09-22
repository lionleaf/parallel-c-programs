run : spmv
	./spmv 100000 401 200 100 200 10

spmv : spmv.c
	gcc -Wall -std=c99 -ggdb -fprofile-arcs -ftest-coverage -O3 -msse3 spmv.c -o spmv -lm
	
clean :
	rm spmv
