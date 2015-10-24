
/*
Performance Comparisions (HSA vs pure CPU) for standard CUDA programs
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <time.h>
#include <stdlib.h>
#include <Windows.h>
#include <stdio.h>

float executiontime = 0;
double cputime = 0;

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

//Populates arr of size with random elements
void init_array(float *arr, const int size)
{
	
	int i;
	for (i = 0; i < size; i++)
	{
		arr[i] = (rand()/float(RAND_MAX) );
	}
}

//Prints arr of size
void print_array(float *arr, const int size)
{
	int i;
	for (i = 0; i < size; i++)
	{
		printf("%f ", arr[i]);
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

//Cuda Vector Addition kernel
__global__ void vecAddKernel(float *A, float *B, float *C, int n)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < n)	C[i] = A[i] + B[i];
}

//Cuda Vector Addition
//h_A, h_B are vectors on host memory
//h_C is the result
//
void vecAdd(float *h_A, float *h_B, float *h_C, int n)
{
	int s = n * sizeof(float);
	float *d_A, *d_B, *d_C;

	//Time metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Error checking
	cudaError_t err;

	//Allocate memory on device and copy contents of A and B
	err = cudaMalloc((void **) &d_A, s);
	errcheck(err);
	cudaMemcpy(d_A, h_A, s, cudaMemcpyHostToDevice);
	err = cudaMalloc((void **)&d_B, s);
	errcheck(err);
	cudaMemcpy(d_B, h_B, s, cudaMemcpyHostToDevice);

	//Allocate memory on device for C
	err = cudaMalloc((void **)&d_C, s);
	errcheck(err);

	//Kernel Incovation
	//Set a 1D Array grid for 256 threads
	//(n-1)/265 + 1 is the cealing for no. of blocks
	dim3 DimGrid((n - 1) / 256 + 1, 1, 1);				//no of blocks
	dim3 DimBlock(256, 1, 1);							//no of threads per block

	cudaEventRecord(start);
	vecAddKernel<<<DimGrid, DimBlock>>>(d_A, d_B, d_C, n);
	cudaEventRecord(stop);

	//Copy memory from device to host for C
	cudaMemcpy(h_C, d_C, s, cudaMemcpyDeviceToHost);

	//Syncronize and display time taken
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&executiontime, start, stop);

	//Free memory from Device
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void cpuAdd(float *A, float *B, float *C, float *D, int n)
{
	int i;
	double start, stop;

	start = get_cpu_time();
	for (i = 0; i < n; i++)
	{
		D[i] = A[i] + B[i];
	}
	stop = get_cpu_time();
	cputime = stop - start;

}

int main()
{
	printdevices();
	
	srand(time(NULL));

	//Number of array elements
	//Change depending on memory constraints
	const int size = 99999999;
	
	float *h_A, *h_B, *h_C, *h_D;
	h_A = (float *)malloc(size * sizeof(float));
	h_B = (float *)malloc(size * sizeof(float));
	h_C = (float *)calloc(size, sizeof(float));
	h_D = (float *)malloc(size * sizeof(float));

	printf("\nGenerating random floating point arrays.\n");
	init_array(h_A, size);
	init_array(h_B, size);
	printf("\nGeneration Complete.");

	/*
	printf("Array A:\n");
	print_array(h_A, size);
	printf("\n\nArray B:\n");
	print_array(h_B, size);
	*/

	printf("\n\nStarting CUDA Compute...\n");

	//Execute CUDA Component
	vecAdd(h_A, h_B, h_C, size);

	printf("CUDA Compute Complete. \n\nStarting CPU Compute...");

	//Execute CPU verification
	cpuAdd(h_A, h_B, h_C, h_D, size);

	/*
	printf("Array C: A+B\n");
	print_array(h_C, size);

	printf("\nArray D: A+B\n");
	print_array(h_D, size);
	*/

	printf("\nCPU Compute Complete...\n");

	printf("\n\nCuda Execution time: %f ms", executiontime);
	printf("\n\nCPU Execution time: %f ms", cputime);
	printf("\n\n");
	return 0;
}