#include "header.cuh"

__global__ void naive_kernel(const Matrix d_A, Matrix d_B)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    unsigned int col = bx * blockDim.x + tx;
    unsigned int row = by * blockDim.y + ty;

    if(col < d_B.width && row < d_B.height)
    {
        DATA_TYPE temp;
        temp = __ldg(&d_A.data[col * d_A.width + row]);
        d_B.data[row * d_B.width + col] = temp;
    }

}

__global__ void shared_kernel(const Matrix d_A, Matrix d_B)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    unsigned int col = bx * blockDim.x + tx;
    unsigned int row = by * blockDim.y + ty;

    __shared__ DATA_TYPE s_d[TILE_WIDTH][TILE_WIDTH];

    if(col < d_B.width && row < d_B.height)
    {
        s_d[ty][tx] = __ldg(&d_A.data[col * d_A.width + row]);
        __syncthreads();
    }

    if(col < d_B.width && row < d_B.height)
        d_B.data[row * d_B.width + col] = s_d[ty][tx];

}

__global__ void faster_kernel(const Matrix d_A, Matrix d_B)
{

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    unsigned int col = bx * blockDim.x + tx;
    unsigned int row = by * blockDim.y + ty;

    unsigned int col1 = bx * blockDim.x + ty;
    unsigned int row1 = by * blockDim.y + tx;


    __shared__ DATA_TYPE s_d[TILE_WIDTH][TILE_WIDTH];

    if(col < d_A.width && row < d_A.height)
    {
        s_d[ty][tx] = __ldg(&d_A.data[row * d_A.width + col]);
        __syncthreads();
    }

    if(row1 < d_B.width && col1 < d_B.height)
        d_B.data[col1 * d_B.width + row1] = s_d[tx][ty];

}

void transpose_cpu(Matrix h_A, Matrix temp);
//double GetSystemTime(int init);
double get_time(void);

int main(int argc, char *argv[])
{
    unsigned int oriWidth = M;
    unsigned int oriHeight = N;

    if(argc > 1) CheckCmdLine(argc, argv, &oriWidth, &oriHeight);
    printf("width = %d, height = %d\n", oriWidth, oriHeight);

    //host
    Matrix h_A, h_B;

    h_A.width = oriWidth;
    h_A.height = oriHeight;

    h_B.width = h_A.height;
    h_B.height = h_A.width;

    unsigned int SIZE = h_A.width * h_A.height * sizeof(DATA_TYPE);

    h_A.data = (DATA_TYPE*)malloc(SIZE);
    h_B.data = (DATA_TYPE*)malloc(SIZE);

    if(h_A.data == NULL || h_B.data == NULL)
        printf("Failed to allocate memory space on host");

    memset(h_B.data, 0, SIZE);
    ConstInit(h_A.data, h_A.width, h_A.height);

    Matrix reference;
    reference.width = h_A.height;
    reference.height = h_A.width;
    reference.data = (DATA_TYPE*)malloc(SIZE);

    double time_h = get_time();

    transpose_cpu(h_A, reference);

	time_h = get_time() - time_h;
    printf("CPU time = %e ms\n", time_h*1000);

    //device
    Matrix d_A, d_B;

    d_A.width = h_A.width;
    d_A.height = h_A.height;

    d_B.width = h_B.width;
    d_B.height = h_B.height;

    printf("Allocate %e MB on GPU\n", SIZE / (1024.f*1024.f));

    CheckError(cudaMalloc((void**)&d_A.data, SIZE));
    CheckError(cudaMalloc((void**)&d_B.data, SIZE));

    CheckError(cudaMemcpy(d_A.data, h_A.data, SIZE, cudaMemcpyHostToDevice));

    WarmUpGPU();

    double time_d = get_time();

    int nIter = 20;

    //global memory, NOT Consolidated read global memory, BUT write to global memory is Consolidated
    //dim3 block(256);
    //dim3 grid((d_B.width + block.x - 1) / block.x, (d_B.height + block.y - 1) / block.y);
    // for( int i=0; i < nIter; i++)
    //     naive_kernel<<<grid, block_global>>>(d_A, d_B);
    

    //shared memory, NO bank conflict, and NOT coalesced read global memory
    //It's important for a better performance to use 2-D block size rather than 1-D
    // dim3 block2(TILE_WIDTH, TILE_WIDTH);
    // dim3 grid2((d_B.width + block2.x - 1) / block2.x, (d_B.height + block2.y - 1) / block2.y);   
    // for( int i=0; i < nIter; i++)
    //     shared_kernel<<<grid2, block2>>>(d_A, d_B);

    ////most important step for performance: coalesced read and coalesced write global memory
    dim3 block3(TILE_WIDTH, TILE_WIDTH);
    dim3 grid3((d_A.width + block3.x - 1) / block3.x, (d_A.height + block3.y - 1) / block3.y);
    faster_kernel<<<grid3, block3>>>(d_A, d_B);


    CheckError(cudaGetLastError());
    CheckError(cudaDeviceSynchronize());

    time_d = (get_time() - time_d) / nIter;

    //Floating-point operations per second
    double gigaFlops = (2.0 * d_B.width * d_B.height * 1.0e-9f) / (time_d);

    printf("Performance = %.5f GFlop/s, Kernel time = %e ms, SpeedUP=%f\n", gigaFlops, time_d*1000, time_h/time_d);

    CheckError(cudaMemcpy(h_B.data, d_B.data, SIZE, cudaMemcpyDeviceToHost));

    printf("Checking computed result for correctness:\n ");

    bool correct = true;
    double eps = 1.e-6;

    //for(int i = 0; i < h_B.height * h_B.width; i++)
    for(int i = 0; i < 2 * h_B.width; i++)
    {
            DATA_TYPE rel = h_B.data[i];
            DATA_TYPE ref = reference.data[i];

            double abs_err = fabs(rel - ref);
            double abs_val = fabs(rel);
            double rel_err = abs_err / abs_val / h_A.width;

            if(rel_err > eps)
            {
                printf("Error! Matrix[%d]=%.5f, ref=%f error term is > %E\n", i, rel, ref, eps);
                correct = false;
            }

    }
    printf("%s\n", correct ? "Result = PASS" : "Result = FAIL");

    free(h_A.data);
    free(h_B.data);
    free(reference.data);
    cudaFree(d_A.data);
    cudaFree(d_B.data);
    return 0;
}

double get_time(void)
{
    static LARGE_INTEGER nFreq;
    LARGE_INTEGER nTime;
	static int init;
    double t;

    if(init != 1){
        QueryPerformanceFrequency(&nFreq);
		init = 1;
    }
        
    QueryPerformanceCounter(&nTime);

    t = (double)nTime.QuadPart*1. / (double)nFreq.QuadPart;

    return t;
}

void transpose_cpu(Matrix h_A, Matrix temp)
{
    int w = temp.width;
    int h = temp.height;
    for(int i = 0; i < h; i++)
    {
        for( int j = 0; j < w; j++)
        {
            temp.data[i * w + j] = h_A.data[j * h + i];  
            //printf("%.3f  ", temp.data[i * w + j]);

        }
        //printf("\n");
    }

}