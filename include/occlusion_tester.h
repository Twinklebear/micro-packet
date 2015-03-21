#ifndef OCCLUSION_TESTER_H
#define OCCLUSION_TESTER_H

#include "vec.h"
#include "scene.h"

struct OcclusionTester {
	Ray8 rays;

	/*
	 * Set the occlusion tester to check if there's something in between a and b
	 */
	inline void set_points(const Vec3f_8 &a, const Vec3f_8 &b){
		rays = Ray8{a, (b - a).normalized(), 0.001f, 0.999f};
	}
	/*
	 * Get a mask of point pairs that are occluded in in the scene
	 */
	inline __m256 occluded(const Scene &scene){
		DiffGeom8 dg;
		return scene.intersect(rays, dg);
	}
};

#endif

