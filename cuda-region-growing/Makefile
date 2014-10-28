run: raycast
	./raycast
	
raycast : bmp.cpp raycast.cu
	nvcc bmp.cpp raycast.cu -arch=sm_20 -lcudart -lm -o raycast
