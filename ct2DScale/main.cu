/*
Performance Comparisions (HSA vs pure CPU) for standard CUDA programs

Floating point 2D Matrix Scaling
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <Windows.h>
#include <stdio.h>

float executiontime = 0;
double cputime = 0;

double get_cpu_time();
void init_array(float **, const int, const int);
void print_array(float **, const int, const int);
void errcheck(cudaError_t);
void printdevices();

//CUDA 2D Matrix Scaling kernel
__global__ void scaleKernel(float *d_in, float *d_out, int n, int m, float s)
{
	//Calculate row number of d_in and d_out element thread
	int r = blockIdx.y*blockDim.y + threadIdx.y;

	//Calculate col number of d_in and d_out element thread
	int c = blockIdx.x*blockDim.x + threadIdx.x;

	//perform scaling by scale factor s
	if ((r < m) && (c < n))
	{
		d_out[r*n + c] = s * d_in[r*n + c];
	}
}

int main()
{

	return 0;
}

//Windows CPU Time
double get_cpu_time(){
	FILETIME a, b, c, d;
	if (GetProcessTimes(GetCurrentProcess(), &a, &b, &c, &d) != 0){
		//  Returns total user time.
		//  Can be tweaked to include kernel times as well.
		return
			(double)(d.dwLowDateTime |
			((unsigned long long)d.dwHighDateTime << 32)) * 0.0000001;
	}
	else{
		return 0;
	}
}

//Populates arr of rows, cols with random elements
void init_array(float **arr, const int rows, const int cols)
{

	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			arr[i][j] = (rand() / float(RAND_MAX));
		}
	}
}


//Prints arr of rows, cols
void print_array(float **arr, const int rows, const int cols)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			printf("%f ", arr[i][j]);

		}
		printf("\n");
	}
}

//Cuda error checking
void errcheck(cudaError_t cerr)
{
	if (cerr != cudaSuccess)
	{
		printf("%s in %s at %d\n", cudaGetErrorString(cerr), __FILE__, __LINE__);
		exit(EXIT_FAILURE);
	}
}

//Prints information about cuda devices
void printdevices()
{
	int nDevices;
	cudaError_t err;

	err = cudaGetDeviceCount(&nDevices);
	errcheck(err);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printf("Device Number: %d\n", i);
		printf("  Device name: %s\n", prop.name);
		printf("  Memory Clock Rate (KHz): %d\n",
			prop.memoryClockRate);
		printf("  Memory Bus Width (bits): %d\n",
			prop.memoryBusWidth);
		printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
			2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
	}
}
