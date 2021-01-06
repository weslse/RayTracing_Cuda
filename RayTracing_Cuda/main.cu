#include "Timer.h"
#include "rtcuda.h"
#include "hittable_list.h"
#include "sphere.h"

#include <iostream>

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__constant__ point3 d_origin;
__constant__ vec3 d_horizontal;
__constant__ vec3 d_vertical;
__constant__ vec3 d_lower_left_corner;

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

// Device Functions

__device__ color ray_color(const ray& r, const hittable_list* world) {
	hit_record rec;
	if (world->hit(r, 0.f, infinity, rec)) {
		return 0.5f * (rec.normal + color(1.f, 1.f, 1.f));
	}
	
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5f * (unit_direction.y() + 1.f);

	return (1.f - t) * color(1.f, 1.f, 1.f) + t * color(0.5f, 0.7f, 1.f);
}


// Global Functions

__global__ void createWorld(hittable** d_objList, hittable_list** d_world, const size_t objListSize) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		*(d_objList) = new sphere(point3(0.f, 0.f, -1.f), 0.5f);
		*(d_objList + 1) = new sphere(point3(0.f, -100.5f, -1.f), 100.f);
		*d_world = new hittable_list(d_objList, objListSize);
	}
}

__global__ void destoryWorld(hittable** d_objList, hittable_list** d_world, size_t objListSize) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		(*d_world)->clear();
		delete *(d_objList + 1);
		delete *(d_objList);
	}
}

__global__ void render(vec3 *fb, int imgSize_x, int imgSize_y, hittable_list** world) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= imgSize_x) || (j >= imgSize_y))
		return;

	float u = float(i) / (imgSize_x - 1);
	float v = float(j) / (imgSize_y - 1);
	ray r(d_origin, d_lower_left_corner + u * d_horizontal + v * d_vertical - d_origin);
	color pixel_color = ray_color(r, *world);
	int pixel_index = j * imgSize_x + i;
	fb[pixel_index] = pixel_color;
}


int main() {
	Timer* timer = new Timer();

	// Image
	const auto aspect_ratio = 16.f / 9.f;
	int img_x = 400;
	int img_y = static_cast<int>(img_x / aspect_ratio);

	// World
	size_t objListSize = 2;
	hittable** d_objList = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_objList, sizeof(hittable*) * objListSize));

	hittable_list** d_world = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_world, sizeof(hittable_list*)));;

	createWorld << <1, 1 >> > (d_objList, d_world, objListSize);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());


	// Camera
	auto viewport_height = 2.f;
	auto viewport_width = aspect_ratio * viewport_height;
	auto focal_length = 1.f;

	size_t vec_size = sizeof(vec3);

	point3 origin = point3(0.f, 0.f, 0.f);
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&d_origin, &origin, vec_size));

	vec3 horizontal = vec3(viewport_width, 0.f, 0.f);
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&d_horizontal, &horizontal, vec_size));

	vec3 vertical = vec3(0.f, viewport_height, 0.f);
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&d_vertical, &vertical, vec_size));

	vec3 lower_left_corner = origin - horizontal * 0.5f - vertical * 0.5f - vec3(0.f, 0.f, focal_length);
	checkCudaErrors(cudaMemcpyToSymbol((const void*)&d_lower_left_corner, &lower_left_corner, vec_size));

	// Thread
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
	render << <blocks, threads >> > (fb, img_x, img_y, d_world);
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

	// Free memory
	destoryWorld << <1, 1 >> > (d_objList, d_world, objListSize);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(d_world));
	checkCudaErrors(cudaFree(d_objList));
	checkCudaErrors(cudaFree(fb));
	delete timer;

	return 0;
}