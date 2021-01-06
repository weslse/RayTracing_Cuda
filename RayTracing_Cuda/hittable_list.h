#pragma once

#include "hittable.h"

class hittable_list : public hittable {
public:
	__device__ hittable_list() {}
	__device__ hittable_list(hittable** objs, size_t n) : list_size(n) { objects = objs; }

	__device__ void set(hittable** objList, size_t listSize) { objects = objList; list_size = listSize; }
	__device__ void clear() { objects = nullptr; list_size = 0; }
	__device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const override;

public:
	hittable** objects;
	size_t list_size = 0;
};

__device__ bool hittable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
	hit_record tmp_rec;
	bool hit_anything = false;
	auto closest_so_far = t_max;

	for (size_t i = 0; i < list_size; ++i) {
		if (objects[i]->hit(r, t_min, closest_so_far, tmp_rec)) {
			hit_anything = true;
			closest_so_far = tmp_rec.t;
			rec = tmp_rec;
		}
	}

	return hit_anything;
}