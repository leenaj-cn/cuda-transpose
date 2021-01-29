#pragma once
#include <stdio.h>
#include <string.h>

#include <cuda.h>
//sytem time
#include <windows.h>

typedef float DATA_TYPE;

#define M 10000
#define N 20000

#define TILE_WIDTH 16

typedef struct {
    unsigned int width;
    unsigned int height;
    DATA_TYPE* data;
} Matrix;


//CUDA ERROR API
#include <cstdlib>
#define CheckError(call)  \
do{ \
    const cudaError_t error_code = call;  \
    if(error_code  != cudaSuccess) \
    { \
        printf("CUDA Error:\n"); \
        printf("\tFile:\t%s\n", __FILE__);\
        printf("\tLine:\t%d\n", __LINE__);\
        printf("\tError code:%d\n", error_code);\
        printf("\tError info:%s\n", cudaGetErrorString(error_code)); \
        exit(1); \
    } \
}while(0)

//init matrix
void ConstInit(DATA_TYPE *matrix, unsigned int w, unsigned int h)
{
    unsigned int i,j;
    unsigned int size = w * h;
    for(i=0; i < h; i++)
    {
        for(j=0; j < w; j++)
        {
            matrix[i * w + j] = i*2;
            //printf("%.3f\t", matrix[i * w + j]);
        }
       //printf("\n");
    }
}

//warmup gpu
__global__ void warmup_kernel()
{
    float i = 1.0f;
    float j = 2.0f;
    float t = 0.0f;

    t = i * i + j * j;

}

void WarmUpGPU()
{
    for(int i = 0; i < 20; i++)
    {
        warmup_kernel<<<1, 256>>>();
    
    }
}


//get command line
int CheckCmdLine(int argc, char *argv[], unsigned int *width,unsigned int *height)
{
    printf("argc=%d\n",argc);
    if(argc==5)
    {
        printf("Your Input: ");

        int i;
        for(i=0;i<argc;i++)
        {
            printf("%s ",argv[i]);
        }
        printf("\n");

        *width = atoi(argv[2]);
        *height = atoi(argv[4]);

        printf("width=%d, height=%d\n",*width, *height);
    }
    else// && strcmp(argv[1] ,"--help")
    {
            printf("Please set the width, height of your matrix, ");
            printf("For example: -w width -h height\n");
            exit(-1);
    }
 
    return 0;
}