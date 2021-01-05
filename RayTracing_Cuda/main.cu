#include <iostream>
#include <time.h>

#include "custom_cudart.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__global__ void render(float *fb, int max_x, int max_y) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x * 3 + i * 3;
	fb[pixel_index + 0] = float(i) / max_x;
	fb[pixel_index + 1] = float(j) / max_y;
	fb[pixel_index + 2] = 0.2;
}

int main() {
	int img_x = 256;
	int img_y = 256;
	int tx = 8;
	int ty = 8;

	int num_pixels = nx * ny;
	size_t fb_size = 3 * num_pixels * sizeof(float);

	// allocate FB
	float *fb;
	checkCudaErrors(cudaMalloc((void **)&fb, fb_size));

	// Render our buffer
	dim3 blocks(img_x / tx + 1, img_y / ty + 1);
	dim3 threads(tx, ty);
	render << <blocks, threads >> > (fb, img_x, img_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	// Output FB as Image
	std::cout << "P3\n" << img_x << " " << img_y << "\n255\n";
	for (int j = img_y - 1; j >= 0; j--) {
		for (int i = 0; i < img_x; i++) {
			size_t pixel_idx = j * 3 * img_x + i * 3;
			float r = fb[pixel_idx + 0];
			float g = fb[pixel_idx + 1];
			float b = fb[pixel_idx + 2];
			int ir = int(255.99*r);
			int ig = int(255.99*g);
			int ib = int(255.99*b);
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}

	checkCudaErrors(cudaFree(fb));
}