/*
Performance Comparisions (HSA vs pure CPU) for standard CUDA programs

Floating point 2D Matrix Scaling

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

//CUDA Kernel Init and Call
void ctScale(float *h_in, float *h_out, int n, int m, float s)
{
	//Initialize device var
	cudaError_t err;
	float *d_in, *d_out;
	int dsize = n * m * sizeof(float);

	//Time metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Device memory allocation for input and copy
	err = cudaMalloc((void **)&d_in, dsize);
	errcheck(err);
	cudaMemcpy(d_in, h_in, dsize, cudaMemcpyHostToDevice);

	//Device memory allocation for output
	err = cudaMalloc((void **)&d_out, dsize);
	errcheck(err);

	//Kernel invoke, assume 2D Grids (size 16) each with 16 threads
	dim3 DimGrid((n - 1) / 16 + 1, (m - 1) / 16 + 1, 1);
	dim3 DimBlock(16, 16, 1);

	cudaEventRecord(start);
	scaleKernel<<<DimGrid,DimBlock>>>(d_in, d_out, n, m, s);
	cudaEventRecord(stop);

	//Copy output to host
	cudaMemcpy(h_out, d_out, dsize, cudaMemcpyDeviceToHost);

	//Syncronize and display time taken
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&executiontime, start, stop);

	//Free memory from Device
	cudaFree(d_in);
	cudaFree(d_out);
}

//CPU Calculation
void cpuScale(float *h_in, float *h_out, int n, int m, float s)
{
	int i, j;
	double start, stop, wstart, wstop;

	wstart = get_wall_time();
	start = get_cpu_time();
	for (i = 0; i < n; i++)
	{
		for (j = 0; j < m; j++)
		{
			h_out[i*m + j] = s * h_in[i*m + j];
		}
	}
	stop = get_cpu_time();
	wstop = get_wall_time();

	cputime = stop - start;
	walltime = wstop - wstart;
}

int main()
{
	//Print Device
	printdevices();

	//Row/Column size - change depending on memory constraints
	const int rowsize = 20000;			
	const int colsize = 2000;

	//Scale factor
	const float scale = 2.6548;

	//Initialize Arrays: Allocate memory
	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(rowsize * colsize * sizeof(float));
	h_B = (float *)calloc(rowsize * colsize,  sizeof(float));
	h_C = (float *)calloc(rowsize * colsize,  sizeof(float));

	//Initialize h_A - randomized float elements
	printf("\nGenerating Random float point matrix...");
	init_array(h_A, rowsize, colsize);
	printf("\nGeneration complete.\n");

	/*Optional printing of elements, don't use for large row/col size*/
	/*
	printf("\nA:\n");
	print_array(h_A, rowsize, colsize);
	*/

	//CUDA Calculation
	printf("\n\nStarting CUDA Calculation...");
	ctScale(h_A, h_B, rowsize, colsize, scale);
	printf("\nCUDA Calculation complete.\n");

	/*Optional printing of elements, don't use for large row/col size*/
	/*
	printf("\nB = A * %f:\n", scale);
	print_array(h_B, rowsize, colsize);
	*/

	//CPU Calculation
	printf("\n\nStarting CPU Calculation...");
	cpuScale(h_A, h_C, rowsize, colsize, scale);
	printf("\nCPU Calculation complete.\n");
	
	/*Optional printing of elements, don't use for large row/col size*/
	/*
	printf("\nC = A * %f:\n", scale);
	print_array(h_C, rowsize, colsize);
	*/

	//Display performance comparision:
	printf("\nCUDA Execution time: %f ms", executiontime);
	printf("\nCPU Process Execution time: %f ms", cputime * 1000);
	printf("\nCPU Real Execution time: %f ms", walltime * 1000);

	//Free memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("\n\n");
	return 0;
}

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
