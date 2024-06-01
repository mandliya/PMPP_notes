/*
Code to convert an RGB image to grayscale using CUDA
For educational purposes only
*/

#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>


__global__ void rgb_to_grayscale_kernel(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int i = y * width + x;
        gray[i] = 0.299f * r[i] + 0.587f * g[i] + 0.114f * b[i];
    }
}


void rgb_to_grayscale_gpu(unsigned char* r, unsigned char* g, unsigned char* b, unsigned char* gray, int width, int height) {
    unsigned char* d_r, *d_g, *d_b, *d_gray;
    cudaMalloc(&d_r, width * height * sizeof(unsigned char));
    cudaMalloc(&d_g, width * height * sizeof(unsigned char));
    cudaMalloc(&d_b, width * height * sizeof(unsigned char));
    cudaMalloc(&d_gray, width * height * sizeof(unsigned char));

    cudaMemcpy(d_r, r, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, g, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    rgb_to_grayscale_kernel<<<grid, block>>>(d_r, d_g, d_b, d_gray, width, height);

    cudaMemcpy(gray, d_gray, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_r);
    cudaFree(d_g);
    cudaFree(d_b);
    cudaFree(d_gray);
}

int main() {
    int width = 1920;
    int height = 1080;
    
    // Generate a random image with opencv and save it to disk
    cv::Mat colorImage(height, width, CV_8UC3);
    cv::randu(colorImage, cv::Scalar(0, 0, 0), cv::Scalar(255, 255, 255));
    if (cv::imwrite("color.jpg", colorImage)) {
        std::cout << "Color image saved!" << std::endl;
    } else {
        std::cerr << "Error: Could not save color image" << std::endl;
    }

    // Convert to grayscale using CUDA
    cv::Mat grayImage(height, width, CV_8UC1);
    rgb_to_grayscale_gpu(colorImage.ptr(0), colorImage.ptr(1), colorImage.ptr(2), grayImage.ptr(0), width, height);

    // Save the grayscale image to disk
    if (cv::imwrite("gray.jpg", grayImage)) {
        std::cout << "Gray image saved!" << std::endl;
    } else {
        std::cerr << "Error: Could not save gray image" << std::endl;
    }
}