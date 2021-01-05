#pragma once

#include "vec3.h"

class ray {
public:
	__host__ __device__ ray() {}
	__host__ __device__ ray(const point3& origin, const vec3& direction)
		: org(origin), dir(direction)
	{}

	__host__ __device__ inline point3 origin() const { return org; }
	__host__ __device__ inline point3 direction() const { return dir; }

	__host__ __device__ inline point3 at(float t) const { return org + t * dir; }

public:
	point3 org;
	vec3 dir;
};
