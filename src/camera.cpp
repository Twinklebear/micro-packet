#define _USE_MATH_DEFINES
#include <cmath>
#include "camera.h"

PerspectiveCamera::PerspectiveCamera(Vec3f pos, Vec3f center, Vec3f up, float fovy, float aspect)
	: pos(pos), dir((center - pos).normalized()), up(up)
{
	Vec3f dz = dir.normalized();
	Vec3f dx = -dz.cross(up).normalized();
	Vec3f dy = dx.cross(dz).normalized();
	float dim_y = 2.f * std::sin((fovy / 2.f) * (M_PI / 180.f));
	float dim_x = dim_y * aspect;
	dir_top_left = dz - 0.5f * dim_x * dx - 0.5f * dim_y * dy;
	screen_du = dx * dim_x;
	screen_dv = dy * dim_y;
}
void PerspectiveCamera::generate_rays(Ray8 &rays, const Vec2f_8 &samples) const {
	rays.o = Vec3f_8{pos.x, pos.y, pos.z};
	rays.d = Vec3f_8{dir_top_left.x, dir_top_left.y, dir_top_left.z};
	const auto u_step = samples.x * Vec3f_8{screen_du.x, screen_du.y, screen_du.z};
	const auto v_step = samples.y * Vec3f_8{screen_dv.x, screen_dv.y, screen_dv.z};
	rays.d = rays.d + u_step + v_step;
	rays.d.normalize();
	rays.t_min = _mm256_set1_ps(0);
	rays.t_max = _mm256_set1_ps(INFINITY);
}

