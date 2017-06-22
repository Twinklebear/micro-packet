#ifndef CAMERA_H
#define CAMERA_H

#include "vec.h"

/*
 * Simple perspective camera. Perhaps later switch to have transformation matrices?
 * would make it easier to have interactive rendering and could add in my glt arball camera
 */
struct PerspectiveCamera {
	// dir_top_left is the direction from the camera to the top left of the image
	Vec3f pos, dir, up, dir_top_left, screen_du, screen_dv;

	PerspectiveCamera(Vec3f pos, Vec3f center, Vec3f up, float fovy, float aspect);
	/*
	 * Generate a ray packet sampling the 8 screen positions passed,
	 * screen positions should be normalized to be between [0, 1] in
	 * each dimension
	 */
	void generate_rays(RayN &rays, const Vec2fN &samples) const;
};

#endif

