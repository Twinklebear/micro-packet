#ifndef OCCLUSION_TESTER_H
#define OCCLUSION_TESTER_H

#include "vec.h"
#include "scene.h"

struct OcclusionTester {
	RayN rays;

	/*
	 * Set the occlusion tester to check if there's something in between a and b
	 */
	inline void set_points(const Vec3fN &a, const Vec3fN &b){
		rays = RayN{a, (b - a).normalized(), 0.001f, 0.999f};
	}
	/*
	 * Get a mask of point pairs that are occluded in in the scene
	 */
	inline psimd::mask<> occluded(const Scene &scene){
		DiffGeomN dg;
		return scene.intersect(rays, dg) && rays.active;
	}
};

#endif

