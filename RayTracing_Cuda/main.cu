#include <iostream>
#include <time.h>

#include "Timer.h"
#include "vec3.h"
#include "ray.h"


// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

__constant__ point3 d_origin = { 0,0,0 };

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
	if (result) {
		std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}

__device__ bool hit_sphere(const point3& center, float radius, const ray& r) {
	vec3 oc = r.origin() - center;
	auto a = dot(r.direction(), r.direction());
	auto b = 2.f * dot(oc, r.direction());
	auto c = dot(oc, oc) - radius * radius;
	auto discriminant = b * b - 4 * a * c;
	return (discriminant > 0.f);
}

__device__ color ray_color(const ray& r) {
	if (hit_sphere(point3(0.f, 0.f, -1.f), 0.5f, r))
		return color(1.f, 0.f, 0.f);
	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5f * (unit_direction.y() + 1.f);
	return (1.f - t) * color(1.f, 1.f, 1.f) + t * color(0.5f, 0.7f, 1.f);
}

__global__ void render(vec3 *fb, int max_x, int max_y,
	point3* origin, vec3* horizontal, vec3* vertical, vec3* lower_left_corner) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if ((i >= max_x) || (j >= max_y)) return;
	float u = float(i) / (max_x - 1);
	float v = float(j) / (max_y - 1);
	ray r(*origin, *lower_left_corner + u * *horizontal + v * *vertical - *origin);
	color pixel_color = ray_color(r);
	int pixel_index = j * max_x + i;
	fb[pixel_index] = pixel_color;
}

int main() {
	Timer* timer = new Timer();

	// Image
	const auto aspect_ratio = 16.f / 9.f;
	int img_x = 400;
	int img_y = static_cast<int>(img_x / aspect_ratio);

	// Camera

	auto viewport_height = 2.f;
	auto viewport_width = aspect_ratio * viewport_height;
	auto focal_length = 1.f;

	size_t vec_size = sizeof(vec3);

	point3* origin = nullptr;
	checkCudaErrors(cudaMallocManaged((void **)&origin, vec_size));
	*origin = point3(0.f, 0.f, 0.f);

	vec3* horizontal = nullptr;
	checkCudaErrors(cudaMallocManaged((void **)&horizontal, vec_size));
	*horizontal = vec3(viewport_width, 0.f, 0.f);

	vec3* vertical = nullptr;
	checkCudaErrors(cudaMallocManaged((void **)&vertical, vec_size));
	*vertical = vec3(0.f, viewport_height, 0.f);

	vec3* lower_left_corner = nullptr;
	checkCudaErrors(cudaMallocManaged((void **)&lower_left_corner, vec_size));
	*lower_left_corner = *origin - *horizontal * 0.5f - *vertical * 0.5f - vec3(0.f, 0.f, focal_length);


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
	render << <blocks, threads >> > (fb, img_x, img_y, origin, horizontal, vertical, lower_left_corner);
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