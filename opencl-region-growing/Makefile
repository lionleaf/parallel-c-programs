run: raycast
	./raycast
	
raycast : bmp.c clutil.c raycast.c
	gcc -std=c99 bmp.c clutil.c raycast.c -lOpenCL -lm -o raycast
