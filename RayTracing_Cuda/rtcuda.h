#pragma once

// Common Headers
#include "ray.h"
#include "vec3.h"

// Constants
__constant__ float infinity = std::numeric_limits<float>::infinity();
__constant__ float pi = 3.141592f;

// Utility Functions
__device__ inline float degrees_to_radians(float degrees) {
	return degrees * pi / 180.f;
}
