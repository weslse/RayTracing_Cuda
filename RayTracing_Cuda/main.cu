#include <iostream>
#include <time.h>

#include "Timer.h"
#include "vec3.h"

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

__global__ void render(vec3 *fb, int max_x, int max_y) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	fb[pixel_index] = color(float(i) / max_x, float(j) / max_y, 0.2f);
}

int main() {
	Timer* timer = new Timer();

	int img_x = 256;
	int img_y = 256;
	int tx = 8;
	int ty = 8;
	
	std::cerr << "Rendering a " << img_x << "x" << img_y << " image ";
	std::cerr << "in " << tx << "x" << ty << " blocks.\n";

	int num_pixels = img_x * img_y;
	size_t fb_size = num_pixels * sizeof(vec3);

	// allocate FB
	vec3 *fb;
	checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

	// Render our buffer
	dim3 blocks(img_x / tx + 1, img_y / ty + 1);
	dim3 threads(tx, ty);

	timer->Start();
	render << <blocks, threads >> > (fb, img_x, img_y);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	
	timer->End();
	std::cerr << "took " << timer->GetElapsedTime() << " seconds.\n";

	// Output FB as Image
	std::cout << "P3\n" << img_x << " " << img_y << "\n255\n";

	for (int j = img_y - 1; j >= 0; --j) {
		for (int i = 0; i < img_x; ++i) {
			size_t pixel_idx = j * img_x + i;
			float r = fb[pixel_idx + 0].r();
			float g = fb[pixel_idx + 1].g();
			float b = fb[pixel_idx + 2].b();
			int ir = int(255.999f * r);
			int ig = int(255.999f * g);
			int ib = int(255.999f * b);
			std::cout << ir << " " << ig << " " << ib << "\n";
		}
	}
	
	checkCudaErrors(cudaFree(fb));
	delete timer;

	return 0;
}