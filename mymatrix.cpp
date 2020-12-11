#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>

//定义结构体
typedef struct{
        int width;
        int height;
        int stride;//定义矩阵的步长
        float* elements;
}Matrix;

#define BLOCK_SIZE 16

//初始化
void initial(float* A,int N)
{
     int i;
     for(i = 0;i<N;i++)
     {
        A[i] = rand()%10; //随机生成一位数组内的数值，N为大小
     }
}

//获取数组对应下标的值
__device__ float GetElement(const Matrix A,int row,int col)
{
        return A.elements[row*A.stride+col];
}
//设置相应的值
__device__ void SetElement(Matrix A,int row,int col,float value)
{
        A.elements[row*A.stride+col]=value;
}
//得到分块矩阵
__device__ Matrix GetSubMatrix(Matrix A,int row,int col)
{
        Matrix Asub;
        Asub.width = BLOCK_SIZE;
        Asub.height = BLOCK_SIZE;
        Asub.stride = A.stride;
        Asub.elements = &A.elements[A.stride*BLOCK_SIZE*row+BLOCK_SIZE*col];// 得到对应的值
        return Asub;
}

//核函数
__global__ void MatMulKernel(Matrix A,Matrix B,Matrix C)
{
       int blockRow = hipBlockIdx_y; //获取block的行数
       int blockCol = hipBlockIdx_x;
       Matrix Csub = GetSubMatrix(C,blockRow,blockCol);//对C分块
       float Cvalue = 0;
       int row = hipThreadIdx_y;
       int col = hipThreadIdx_x;
       for(int m=0; m<(A.width/BLOCK_SIZE);++m)
       {
        Matrix Asub = GetSubMatrix(A,blockRow,m);
        Matrix Bsub = GetSubMatrix(B,m,blockCol);
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[row][col]=GetElement(Asub,row,col);
        Bs[row][col]=GetElement(Bsub,row,col);

        __syncthreads();
        for(int e = 0;e<BLOCK_SIZE;++e)
        {
                Cvalue += As[row][e]*Bs[e][col];
        }
        __syncthreads();
        SetElement(Csub,row,col,Cvalue);
       }
}
//
void MatMul(const Matrix A,const Matrix B,Matrix C)
{
        Matrix d_A;
        d_A.width = d_A.stride = A.width;
        d_A.height = A.height;
        size_t size = A.width * A.height * sizeof(float);
        //在GPU上申请空间
        hipMalloc(&d_A.elements,size);
        //将矩阵从CPU复制到GPU
        hipMemcpy(d_A.elements,A.elements,size,hipMemcpyHostToDevice);

        Matrix d_B;
        d_B.width = d_B.stride=B.width;
        d_B.height = B.height;
        size = B.width * B.height * sizeof(float);
        hipMalloc(&d_B.elements,size);
        hipMemcpy(d_B.elements,B.elements,size,hipMemcpyHostToDevice);

        Matrix d_C;
        d_C.width = d_C.stride =  C.width;
        d_C.height = C.height;
        size = C.width * C.height * sizeof(float);
        //在GPU上申请内存
        hipMalloc(&d_C.elements,size);
        dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        dim3 dimGrid(B.width / dimBlock.x,A.height / dimBlock.y);

        float gpu_time;
        hipEvent_t start_GPU,stop_GPU;
        hipEventCreate(&start_GPU);
        hipEventCreate(&stop_GPU);
        hipEventRecord(start_GPU,0);

        hipLaunchKernelGGL(MatMulKernel,dimGrid,dimBlock,0,0,d_A,d_B,d_C);

        hipEventRecord(stop_GPU,0);
        hipEventSynchronize(start_GPU);
        hipEventSynchronize(stop_GPU);
        hipEventElapsedTime(&gpu_time,start_GPU,stop_GPU);
        hipDeviceSynchronize();
        printf("\nGPU spend time is: %lf(s)\n",gpu_time/1000);
        hipEventDestroy(start_GPU);
        hipEventDestroy(stop_GPU);

        hipMemcpy(C.elements,d_C.elements,size,hipMemcpyDeviceToHost);

       

        hipFree(d_A.elements);
        hipFree(d_B.elements);
        hipFree(d_C.elements);
}


int main()
{
        Matrix A;
        Matrix B;
        Matrix C;

        A.width = 5760;
        A.height = 5760;
        B.width = 5760;
        B.height = 5760;
        C.width = 5760;
        C.height = 5760;

        int size_A = A.width * A.height * sizeof(float);
        int size_B = B.width * B.height * sizeof(float);
        int size_C = C.width * C.height * sizeof(float);

        A.elements = (float *)malloc(size_A);
        B.elements = (float *)malloc(size_B);
        C.elements = (float *)malloc(size_C);

        initial(A.elements,A.height*A.width);
       
        
        initial(B.elements,B.height*B.width);

        MatMul(A,B,C);
        return 0;
}