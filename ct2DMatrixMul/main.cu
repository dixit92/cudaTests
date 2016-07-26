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

//Default - 16 threads in a block, 16 blocks in a grid
#define TILEWIDTH 16

float executiontime = 0;
double cputime = 0;
double walltime = 0;

double get_cpu_time();
double get_wall_time();
void init_array(float *, const int, const int);
void print_array(float *, const int, const int);
void errcheck(cudaError_t);
void printdevices();

//CUDA kernel for shared memory tiled multiply (FRAMEWORK)
__global__ void MatMulTiledKernel(float *A, float *B, float *C, int m, int n, int k)
{
	//Shared memory for tiled MatMul 
	//Shared memory on GPU is much faster than GPU global mem
	__shared__ float ds_A[TILEWIDTH][TILEWIDTH];
	__shared__ float ds_B[TILEWIDTH][TILEWIDTH];

}

//Kernel Invocation function for tiled/shared memory multiply (FRAMEWORK)
void ctTiledMatrixMul(float *h_A, float *h_B, float *h_C, int m, int n, int k)
{

}

//CUDA Kernel for simple non-shared multiply
__global__ void MatMulKernel(float *A, float *B, float *C, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	if ((row < m) && (col < k))
	{
		float cvalue = 0.0;

		for (int i = 0; i < n; ++i)
		{
			cvalue = cvalue + A[row*n + i] * B[col + i*k];
		}

		C[row*k + col] = cvalue;
	}
}

//Kernel Invocation function for simple CUDA multiply
void ct2DMatrixMul(float *h_A, float *h_B, float *h_C, int m, int n, int k)
{
	//Initialize device var
	cudaError_t err;
	float *d_A, *d_B, *d_C;

	//Time metrics
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Device memory allocation 
	err = cudaMalloc((void **)&d_A, m * n * sizeof(float));
	errcheck(err);
	cudaMemcpy(d_A, h_A, m * n * sizeof(float), cudaMemcpyHostToDevice);

	err = cudaMalloc((void **)&d_B, n * k * sizeof(float));
	errcheck(err);
	cudaMemcpy(d_B, h_B, n * k * sizeof(float), cudaMemcpyHostToDevice);

	//Device memory allocation for output
	err = cudaMalloc((void **)&d_C, m * k * sizeof(float));
	errcheck(err);

	//Kernel invoke, assume 2D Grids (size 16) each with 16 threads
	dim3 DimGrid((k - 1) / TILEWIDTH + 1, (m - 1) / TILEWIDTH + 1, 1);
	dim3 DimBlock(TILEWIDTH, TILEWIDTH, 1);

	cudaEventRecord(start);
	MatMulKernel <<<DimGrid, DimBlock >>>(d_A, d_B, d_C, m, n, k);
	cudaEventRecord(stop);

	//Copy output to host
	cudaMemcpy(h_C, d_C, m * k * sizeof(float), cudaMemcpyDeviceToHost);

	//Syncronize and display time taken
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&executiontime, start, stop);

	//Free memory from Device
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

//CPU-based implementation of Naive (O(n^3) Matrix Multiplication)
void cpuMatMul(float *A, float *B, float *C, int m, int n, int k)
{
	//Timers
	double start, stop, wstart, wstop;

	//Being Matmul
	wstart = get_wall_time();
	start = get_cpu_time();

	for (int row = 0; row < m; row++)
	{
		for (int col = 0; col < k; col++)
		{
			float sum = 0;

			for (int i = 0; i < n; i++)
			{
				float a = A[row*n + i];
				float b = B[col + i*k];
				sum = sum + a * b; 
			}

			C[row*k + col] = sum;
		}
	}

	stop = get_cpu_time();
	wstop = get_wall_time();
	//end Matmul

	//calculate time
	cputime = stop - start;
	walltime = wstop - wstart;
}

int main()
{
	//Print Device
	printdevices();

	///*Optional printing of elements, don't use for large row/col size*/
	bool PrintEle = false;

	//Row/Column size - change depending on memory constraints
	//We assume multiplication of Matrices A and B with a resultant C (for CUDA) and D (for CPU)
	//A = m x n (rows x cols)
	//B = n x k (rows x cols)
	//C = m x k (rows x cols) (result of multiplying A and B)

	const int m = 5000;			
	const int n = 2000;
	const int k = 3000;

	//Initialize Arrays: Allocate memory
	float *h_A, *h_B, *h_C, *h_D, *h_T;
	h_A = (float *)malloc(m * n * sizeof(float));
	h_B = (float *)malloc(n * k * sizeof(float));
	h_C = (float *)calloc(m * k,  sizeof(float));
	h_D = (float *)calloc(m * k,  sizeof(float));
	h_T = (float *)calloc(m * k,  sizeof(float));

	//Initialize h_A (m x n) - randomized float elements
	printf("\nGenerating Random float point matrix A...");
	init_array(h_A, m, n);
	printf("\nGeneration complete.\n");

	/*Optional printing of elements, don't use for large row/col size*/
	if (PrintEle)
	{
		printf("\nA:\n");
		print_array(h_A, m, n);
	}

	//Initialize h_B (n x k) - randomized float elements
	printf("\nGenerating Random float point matrix B...");
	init_array(h_B, n, k);
	printf("\nGeneration complete.\n");

	/*Optional printing of elements, don't use for large row/col size*/
	if (PrintEle)
	{
		printf("\nB:\n");
		print_array(h_B, n, k);
	}

	
	//CUDA Calculation (Naive)
	printf("\n\nStarting basic CUDA Calculation...");
	ct2DMatrixMul(h_A, h_B, h_C, m, n, k);
	printf("\nCUDA Calculation complete.\n");

	//Optional printing of elements, don't use for large row/col size
	if (PrintEle)
	{
		printf("\nC = A . B:\n");
		print_array(h_C, m, k);
	}
	

	//CUDA Calculation (using CUDA Sharedmem)
	printf("\n\nStarting optimized CUDA Calculation...");
	ctTiledMatrixMul(h_A, h_B, h_T, m, n, k);
	printf("\nCUDA Calculation complete.\n");

	//Optional printing of elements, don't use for large row/col size
	if (PrintEle)
	{
		printf("\nC = A . B:\n");
		print_array(h_T, m, k);
	}

	//CPU Calculation
	printf("\n\nStarting CPU Calculation...");
	cpuMatMul(h_A, h_B, h_D, m, n, k);
	printf("\nCPU Calculation complete.\n");
	
	/*Optional printing of elements, don't use for large row/col size*/
	if (PrintEle)
	{
		printf("\nD = A . B:\n");
		print_array(h_D, m, k);
	}

	//Display performance comparision:
	printf("\nCUDA Execution time: %f ms", executiontime);
	printf("\nCPU Process Execution time: %f ms", cputime * 1000);
	printf("\nCPU Real Execution time: %f ms", walltime * 1000);

	//Free memory
	free(h_A);
	free(h_B);
	free(h_C);

	printf("\n\n");
	system("pause");
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
