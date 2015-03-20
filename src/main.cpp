#define _USE_MATH_DEFINES
#include <cmath>
#include <iostream>
#include <array>
#include <iomanip>
#include <cfloat>
#include "immintrin.h"

std::ostream& operator<<(std::ostream &os, const __m256 &v){
	float *f = (float*)&v;
	os << "{ ";
	for (int i = 0; i < 8; ++i){
		os << f[i];
		if (i < 7){
			os << ", ";
		}
	}
	os << " }";
	return os;
}

template<typename T>
T clamp(T x, T min, T max){
	return x < min ? min : x > max ? max : x;
}

// Attempt to solve the quadratic equation. Returns a mask of successful solutions (0xff)
// and stores the computed t values in t0 and t1
__m256 solve_quadratic(const __m256 a, const __m256 b, const __m256 c, __m256 &t0, __m256 &t1){
	auto discrim = _mm256_sub_ps(_mm256_mul_ps(b, b),
			_mm256_mul_ps(_mm256_mul_ps(a, c), _mm256_set1_ps(4.f)));
	auto solved = _mm256_cmp_ps(discrim, _mm256_set1_ps(0), _CMP_GT_OQ);
	// Test for the case where none of the equations can be solved (eg. none hit)
	if ((_mm256_movemask_ps(solved) & 255) == 0){
		return solved;
	}
	// Compute +/-sqrt(discrim), setting -discrim where we have b < 0
	discrim = _mm256_sqrt_ps(discrim);
	auto neg_discrim = _mm256_mul_ps(discrim, _mm256_set1_ps(-1.f));
	// Blend the discriminants to pick the right +/- value
	// Find mask for this with b < 0 set to 1
	auto mask = _mm256_cmp_ps(b, _mm256_set1_ps(0), _CMP_LT_OQ);
	discrim = _mm256_blendv_ps(discrim, neg_discrim, mask);
	auto q = _mm256_mul_ps(_mm256_set1_ps(-0.5f), _mm256_add_ps(b, discrim));
	auto x = _mm256_div_ps(q, a);
	auto y = _mm256_div_ps(c, q);
	// Find which elements have t0 > t1 and compute mask so we can swap them
	mask = _mm256_cmp_ps(x, y, _CMP_GT_OQ);
	t0 = _mm256_blendv_ps(x, y, mask);
	t1 = _mm256_blendv_ps(y, x, mask);
	return solved;
}

// A single vec3f
struct Vec3f {
	float x, y, z;

	Vec3f(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z){}
	inline float dot(const Vec3f &v) const {
		return x * v.x + y * v.y + z * v.z;
	}
	inline Vec3f cross(const Vec3f &v) const {
		return Vec3f{y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x};
	}
	inline float length_sqr() const {
		return x * x + y * y + z * z;
	}
	inline float length() const {
		return std::sqrt(length_sqr());
	}
	inline Vec3f normalized() const {
		const float len = length();
		return Vec3f{x / len, y / len, z / len};
	}
};
inline Vec3f operator+(const Vec3f &a, const Vec3f &b){
	return Vec3f{a.x + b.x, a.y + b.y,  a.z + b.z};
}
inline Vec3f operator-(const Vec3f &a, const Vec3f &b){
	return Vec3f{a.x - b.x, a.y - b.y,  a.z - b.z};
}
inline Vec3f operator*(const Vec3f &v, float s){
	return Vec3f{v.x * s, v.y * s, v.z * s};
}
inline Vec3f operator*(float s, const Vec3f &v){
	return v * s;
}
inline Vec3f operator/(const Vec3f &v, float s){
	return v * (1 / s);
}
inline Vec3f operator/(const Vec3f &a, const Vec3f &b){
	return Vec3f{a.x / b.x, a.y / b.y, a.z / b.z};
}
inline Vec3f operator-(const Vec3f &v){
	return Vec3f{-v.x, -v.y, -v.z};
}
std::ostream& operator<<(std::ostream &os, const Vec3f &v){
	std::cout << "Vec3f:\n\tx = " << v.x
		<< "\n\ty = " << v.y
		<< "\n\tz = " << v.z;
	return os;
}

// Struct holding 8 vec3f's
struct Vec3f_8 {
	__m256 x, y, z;

	Vec3f_8(float x = 0, float y = 0, float z = 0)
		: x(_mm256_set1_ps(x)), y(_mm256_set1_ps(y)), z(_mm256_set1_ps(z)){}
	Vec3f_8(__m256 x, __m256 y, __m256 z) : x(x), y(y), z(z){}
	// Compute length^2 of all 8 vectors
	inline __m256 length_sqr() const {
		const auto x_sqr = _mm256_mul_ps(x, x);
		const auto y_sqr = _mm256_mul_ps(y, y);
		const auto z_sqr = _mm256_mul_ps(z, z);
		return _mm256_add_ps(x_sqr, _mm256_add_ps(y_sqr, z_sqr));
	}
	// Compute length of all 8 vectors
	inline __m256 length() const {
		return _mm256_sqrt_ps(length_sqr());
	}
	// Normalize all 8 vectors
	inline void normalize(){
		const auto len = length();
		x = _mm256_div_ps(x, len);
		y = _mm256_div_ps(y, len);
		z = _mm256_div_ps(z, len);
	}
	inline __m256 dot(const Vec3f_8 &vb) const {
		const auto a = _mm256_mul_ps(x, vb.x);
		const auto b = _mm256_mul_ps(y, vb.y);
		const auto c = _mm256_mul_ps(z, vb.z);
		return _mm256_add_ps(a, _mm256_add_ps(b, c));
	}
};
inline Vec3f_8 operator+(const Vec3f_8 &a, const Vec3f_8 &b){
	return Vec3f_8{_mm256_add_ps(a.x, b.x), _mm256_add_ps(a.y, b.y), _mm256_add_ps(a.z, b.z)};
}
inline Vec3f_8 operator-(const Vec3f_8 &a, const Vec3f_8 &b){
	return Vec3f_8{_mm256_sub_ps(a.x, b.x), _mm256_sub_ps(a.y, b.y),
		_mm256_sub_ps(a.z, b.z)};
}
// Scale all components of the vector
inline Vec3f_8 operator*(__m256 s, const Vec3f_8 &v){
	return Vec3f_8{_mm256_mul_ps(s, v.x), _mm256_mul_ps(s, v.y), _mm256_mul_ps(s, v.z)};
}
std::ostream& operator<<(std::ostream &os, const Vec3f_8 &v){
	os << "Vec3f_8:\n\tx = " << v.x
		<< "\n\ty = " << v.y
		<< "\n\tz = " << v.z;
	return os;
}

// Struct holding 8 vec2i's
struct Vec2f_8 {
	__m256 x, y;

	Vec2f_8(float x = 0, float y = 0) : x(_mm256_set1_ps(x)), y(_mm256_set1_ps(y)){}
	Vec2f_8(__m256 x, __m256 y) : x(x), y(y){}
};
Vec2f_8 operator/(const Vec2f_8 &a, const Vec2f_8 &b){
	return Vec2f_8{_mm256_div_ps(a.x, b.x), _mm256_div_ps(a.y, b.y)};
}
std::ostream& operator<<(std::ostream &os, const Vec2f_8 &v){
	os << "Vec2f_8:\n\tx = " << v.x
		<< "\n\ty = " << v.y;
	return os;
}


// Packet of 8 rays
struct Ray8 {
	Vec3f_8 o, d;
	__m256 t_max, t_min;

	Ray8(Vec3f_8 o = Vec3f_8{}, Vec3f_8 d = Vec3f_8{}) : o(o), d(d), t_max(_mm256_set1_ps(INFINITY)),
		t_min(_mm256_set1_ps(0))
	{}
};
std::ostream& operator<<(std::ostream &os, const Ray8 &r){
	os << "Ray8:\no = " << r.o
		<< "\nd = " << r.d
		<< "\nt_max = " << r.t_max
		<< "\nt_min = " << r.t_min;
	return os;
}

struct Sphere {
	float x, y, z, radius;

	Sphere(float x, float y, float z, float radius) : x(x), y(y), z(z), radius(radius) {}
	// Test 8 rays against the sphere, returns masks for the hits (0xff) and misses (0x00)
	__m256 intersect(Ray8 &ray) const {
		const auto center = Vec3f_8{x, y, z};
		const auto d = center - ray.o;
		const auto a = ray.d.length_sqr();
		const auto b = _mm256_mul_ps(ray.d.dot(d), _mm256_set1_ps(-2.f));
		const auto c = _mm256_sub_ps(d.dot(d), _mm256_mul_ps(_mm256_set1_ps(radius),
					_mm256_set1_ps(radius)));
		// Solve the quadratic equation and store the mask of potential hits
		// We'll update this mask as we discard other potential hits, eg. due to
		// the hit being beyond the ray's t value
		std::cout << "a = " << a
			<< "\nb = " << b
			<< "\nc = " << c << std::endl;
		__m256 t0, t1;
		auto hits = solve_quadratic(a, b, c, t0, t1);
		std::cout << "solved quadratic:"
			<< "\nhits = " << hits
			<< "\nt0 = " << t0
			<< "\nt1 = " << t1 << std::endl;
		// We want t0 to hold the nearest t value that is greater than ray.t_min
		auto swap_t = _mm256_cmp_ps(t0, ray.t_min, _CMP_LT_OQ);
		t0 = _mm256_blendv_ps(t0, t1, swap_t);
		// Check which rays are within the ray's t range
		auto in_range = _mm256_and_ps(_mm256_cmp_ps(t0, ray.t_min, _CMP_GT_OQ),
				_mm256_cmp_ps(t0, ray.t_max, _CMP_LT_OQ));
		hits = _mm256_and_ps(hits, in_range);
		// Check if all rays miss the sphere
		if ((_mm256_movemask_ps(hits) & 255) == 0){
			return hits;
		}
		// Update t values for rays that did hit
		ray.t_max = _mm256_blendv_ps(ray.t_max, t0, hits);
		return hits;
	}
};

// Simple perspective camera. Perhaps later switch to have transformation matrices?
// would make it easier to have interactive rendering and could add in my glt arball camera
struct PerspectiveCamera {
	// dir_top_left is the direction from the camera to the top left of the image
	Vec3f pos, dir, up, dir_top_left, screen_du, screen_dv;

	PerspectiveCamera(Vec3f pos, Vec3f center, Vec3f up, float fovy, float aspect)
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
		std::cout << "dir_top_left = " << dir_top_left
			<< "\nscreen_du = " << screen_du
			<< "\nscreen_dv = " << screen_dv
			<< "\ndx = " << dx
			<< "\ndim_x = " << dim_x
			<< "\ndy = " << dy
			<< "\ndim_y = " << dim_y << "\n";
	}
	// Generate a ray packet sampling the 8 screen positions passed
	void generate_rays(Ray8 &rays, const Vec2f_8 &samples) const {
		rays.o = Vec3f_8{pos.x, pos.y, pos.z};
		rays.d = Vec3f_8{dir_top_left.x, dir_top_left.y, dir_top_left.z};
		const auto u_step = samples.x * Vec3f_8{screen_du.x, screen_du.y, screen_du.z};
		const auto v_step = samples.y * Vec3f_8{screen_dv.x, screen_dv.y, screen_dv.z};
		std::cout << "u_step = " << u_step
			<< "\nv_step = " << v_step
			<< "\nrays.d = " << rays.d << "\n";
		rays.d = rays.d + u_step + v_step;
		std::cout << "final (not normalized ray dir) = " << rays.d << "\n";
		rays.d.normalize();
		rays.t_min = _mm256_set1_ps(0);
		rays.t_max = _mm256_set1_ps(INFINITY);
	}
};

int main(int, char**){
	const auto sphere = Sphere{0, 0, 0, 1.25};
	const auto camera = PerspectiveCamera{Vec3f{0, 0, -3}, Vec3f{0, 0, 0}, Vec3f{0, 1, 0}, 60.f, 1.f};
	std::array<float, 16> pixel_x, pixel_y;
	for (int i = 0; i < 16; ++i){
		pixel_x[i] = 0.5 + i % 4;
		pixel_y[i] = 0.5 + i / 4;
	}
	const auto img_dim = Vec2f_8{4, 4};
	const auto samples_top = Vec2f_8{_mm256_loadu_ps(pixel_x.data()),
		_mm256_loadu_ps(pixel_y.data())};
	const auto samples_bot = Vec2f_8{_mm256_loadu_ps(pixel_x.data() + 8),
		_mm256_loadu_ps(pixel_y.data() + 8)};
	std::cout << "samples_top = " << samples_top
	   << "\nsamples_bot = " << samples_bot << "\n";
	// packet_top tests the upper 4x2 region while packet_bot checks the lower 4x2 region
	Ray8 packet_top, packet_bot;
	std::cout << "Making ray packet for top 4x2 region\n";
	camera.generate_rays(packet_top, samples_top / img_dim);
	std::cout << "Making ray packet for bottom 4x2 region\n";
	camera.generate_rays(packet_bot, samples_bot / img_dim);
	std::cout << "packet_top = " << packet_top
	   << "\npacket_bot = " << packet_bot << std::endl;
	// We're rendering a 4x4 'image'
	std::array<char, 4 * 4> image;
	image.fill(' ');
	{
		std::cout << "Checking top hits\n";
		auto hits = sphere.intersect(packet_top);
		std::cout << "Actual hits = " << hits << std::endl;
		std::cout << "packet post hit: " << packet_top << std::endl;

		int hit_mask = _mm256_movemask_ps(hits);
		std::cout << "hit mask = " << std::hex << hit_mask << std::dec << std::endl;
		for (int mask = 1, i = 0; i < 8; mask <<= 1, ++i){
			std::cout << "checking for hit at index " << i
				<< ", using mask " << std::hex << mask << std::dec << std::endl;
			if (hit_mask & mask){
				const float *x = (const float*)&samples_top.x;
				const float *y = (const float*)&samples_top.y;
				std::cout << "there was a hit for pixel { " << x[i] << ", " << y[i] << " }\n";
				const int px_x = clamp(x[i], 0.f, 4.f);
				const int px_y = clamp(y[i], 0.f, 4.f);
				image[px_y * 4 + px_x] = 'X';
			}
		}
	}
	{
		std::cout << "Checking bot hits\n";
		auto hits = sphere.intersect(packet_bot);
		std::cout << "Actual hits = " << hits << std::endl;
		std::cout << "packet post hit: " << packet_bot << std::endl;

		int hit_mask = _mm256_movemask_ps(hits);
		std::cout << "hit mask = " << std::hex << hit_mask << std::dec << std::endl;
		for (int mask = 1, i = 0; i < 8; mask <<= 1, ++i){
			std::cout << "checking for hit at index " << i
				<< ", using mask " << std::hex << mask << std::dec << std::endl;
			if (hit_mask & mask){
				const float *x = (const float*)&samples_bot.x;
				const float *y = (const float*)&samples_bot.y;
				std::cout << "there was a hit for pixel { " << x[i] << ", " << y[i] << " }\n";
				const int px_x = clamp(x[i], 0.f, 4.f);
				const int px_y = clamp(y[i], 0.f, 4.f);
				image[px_y * 4 + px_x] = 'X';
			}
		}
	}
	for (int y = 0; y < 4; ++y){
		for (int x = 0; x < 4; ++x){
			std::cout << image[y * 4 + x];
		}
		std::cout << "\n";
	}
}

