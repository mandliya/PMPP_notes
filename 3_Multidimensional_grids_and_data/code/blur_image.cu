#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

const int BLUR_SIZE = 1;

__global__ void blur_image_kernel(unsigned char* input, unsigned char* output, int width, int height) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < height && col < width) {
        int count = 0;
        int sum = 0;
        for (int i = -BLUR_SIZE; i <= BLUR_SIZE; ++i) {
            for (int j = -BLUR_SIZE; j <= BLUR_SIZE; ++j) {
                int curRow = row + i;
                int curCol = col + j;
                if (curRow >= 0 && curRow < height && curCol >= 0 && curCol < width) {
                    sum += input[curRow * width + curCol];
                    count++;
                }
            }
        }
        output[row * width + col] = sum / count;
    }
}

void blur_image_gpu(unsigned char* input, unsigned char* output, int width, int height) {
    unsigned char* d_input, *d_output;
    cudaMalloc(&d_input, width * height * sizeof(unsigned char));
    cudaMalloc(&d_output, width * height * sizeof(unsigned char));

    cudaMemcpy(d_input, input, width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    blur_image_kernel<<<grid, block>>>(d_input, d_output, width, height);

    cudaMemcpy(output, d_output, width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

int main() {
    // Generate a random black and white image with opencv and save it to disk
    int width = 400;
    int height = 400;
    cv::Mat toBlurImage(height, width, CV_8UC1);
    cv::randu(toBlurImage, cv::Scalar(0), cv::Scalar(255));
    if (cv::imwrite("to_blur.jpg", toBlurImage)) {
        std::cout << "Image saved!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image" << std::endl;
    }

    // Allocate image
    cv::Mat blurredImage(height, width, CV_8UC1);

    // Blur the image using CUDA
    blur_image_gpu(toBlurImage.ptr(0), blurredImage.ptr(0), width, height);

    // Save the blurred image to disk
    if (cv::imwrite("blurred_image.jpg", blurredImage)) {
        std::cout << "Blurred image saved!" << std::endl;
    } else {
        std::cerr << "Error: Could not save image" << std::endl;
    }

}