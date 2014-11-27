__kernel void multiply(__global float *a, __global float* b, __global float* result) {
	int id = get_global_id(0);
	result[id] = a[id] * b[id];
}
