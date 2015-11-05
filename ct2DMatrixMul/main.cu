/*
Performance Comparisions (HSA vs pure CPU) for standard CUDA programs

Floating point 2D Matrix Multiplication

Naive CPU MatMul without optimizations used for comparision

Uses flat array structure
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <Windows.h>
#include <stdio.h>

float executiontime = 0;
double cputime = 0;
double walltime = 0;

double get_cpu_time();
double get_wall_time();
void init_array(float *, const int, const int);
void print_array(float *, const int, const int);
void errcheck(cudaError_t);
void printdevices();



//Windows Wall Time
double get_wall_time(){
	LARGE_INTEGER time, freq;
	if (!QueryPerformanceFrequency(&freq)){
		//  Handle error
		return 0;
	}
	if (!QueryPerformanceCounter(&time)){
		//  Handle error
		return 0;
	}
	return (double)time.QuadPart / freq.QuadPart;
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
void init_array(float *arr, const int rows, const int cols)
{

	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			arr[i*cols + j] = (rand() / float(RAND_MAX));
		}
	}
}


//Prints arr of rows, cols
void print_array(float *arr, const int rows, const int cols)
{
	int i, j;
	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++)
		{
			printf("A[%d][%d]: %f ", i, j, arr[i*cols + j]);

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
