#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include <memory>
#include "immintrin.h"
#include "vec.h"
#include "diff_geom.h"
#include "geometry.h"
#include "material.h"
#include "light.h"

struct Scene {
	std::vector<std::shared_ptr<Geometry>> geometry;
	std::vector<std::shared_ptr<Material>> materials;
	PointLight light;

	Scene(std::vector<std::shared_ptr<Geometry>> geom, std::vector<std::shared_ptr<Material>> mats,
		PointLight light);
	/*
	 * Compute the intersection of the ray packet with the scene
	 * returns mask of rays that hit something
	 */
	__m256 intersect(Ray8 &rays, DiffGeom8 &dg) const;
};

#endif

