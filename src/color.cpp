#include <array>
#include <algorithm>
#include <ostream>
#include "vec.h"
#include "color.h"

Colorf::Colorf(float c) : r(c), g(c), b(c){}
Colorf::Colorf(float r, float g, float b) : r(r), g(g), b(b){}
void Colorf::normalize(){
	r = clamp(r, 0.f, 1.f);
	g = clamp(g, 0.f, 1.f);
	b = clamp(b, 0.f, 1.f);
}
Colorf Colorf::normalized() const {
	return Colorf{clamp(r, 0.f, 1.f), clamp(g, 0.f, 1.f), clamp(b, 0.f, 1.f)};
}
float Colorf::luminance() const {
	return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
bool Colorf::is_black() const {
	return r == 0 && g == 0 && b == 0;
}
Colorf Colorf::to_sRGB() const {
	const float a = 0.055;
	const float b = 1.f / 2.4f;
	Colorf srgb;
	for (int i = 0; i < 3; ++i){
		if ((*this)[i] <= 0.0031308){
			srgb[i] = 12.92 * (*this)[i];
		}
		else {
			srgb[i] = (1 + a) * std::pow((*this)[i], b) - a;
		}
	}
	return srgb;
}
Colorf& Colorf::operator+=(const Colorf &c){
	r += c.r;
	g += c.g;
	b += c.b;
	return *this;
}
Colorf& Colorf::operator-=(const Colorf &c){
	r -= c.r;
	g -= c.g;
	b -= c.b;
	return *this;
}
Colorf& Colorf::operator*=(const Colorf &c){
	r *= c.r;
	g *= c.g;
	b *= c.b;
	return *this;
}
Colorf& Colorf::operator*=(float s){
	r *= s;
	g *= s;
	b *= s;
	return *this;
}
Colorf& Colorf::operator/=(float s){
	return *this *= 1.f / s;
}
float& Colorf::operator[](int i){
	switch (i){
		case 0:
			return r;
		case 1:
			return g;
		default:
			return b;
	}
}
const float& Colorf::operator[](int i) const {
	switch (i){
		case 0:
			return r;
		case 1:
			return g;
		default:
			return b;
	}
}
bool Colorf::has_nans() const {
	return std::isnan(r) || std::isnan(g) || std::isnan(b);
}
Colorf Colorf::exp() const {
	return Colorf{std::exp(r), std::exp(g), std::exp(b)};
}
Colorf::operator Color24() const {
	return Color24(static_cast<uint8_t>(r * 255), static_cast<uint8_t>(g * 255),
		static_cast<uint8_t>(b * 255));
}
Colorf operator+(const Colorf &a, const Colorf &b){
	return Colorf{a.r + b.r, a.g + b.g, a.b + b.b};
}
Colorf operator-(const Colorf &a, const Colorf &b){
	return Colorf{a.r - b.r, a.g - b.g, a.b - b.b};
}
Colorf operator-(const Colorf &c){
	return Colorf{-c.r, -c.g, -c.b};
}
Colorf operator*(const Colorf &a, const Colorf &b){
	return Colorf{a.r * b.r, a.g * b.g, a.b * b.b};
}
Colorf operator*(const Colorf &a, float s){
	return Colorf{a.r * s, a.g * s, a.b * s};
}
Colorf operator*(float s, const Colorf &a){
	return Colorf{a.r * s, a.g * s, a.b * s};
}
Colorf operator/(const Colorf &a, const Colorf &b){
	return Colorf{a.r / b.r, a.g / b.g, a.b / b.b};
}
Colorf operator/(const Colorf &c, float s){
	float inv_s = 1.f / s;
	return c * inv_s;
}
bool operator==(const Colorf &a, const Colorf &b){
	return a.r == b.r && a.g == b.g && a.b == b.b;
}
bool operator!=(const Colorf &a, const Colorf &b){
	return !(a == b);
}
std::ostream& operator<<(std::ostream &os, const Colorf &c){
	os << "Colorf: [r = " << c.r << ", g = " << c.g
		<< ", b = " << c.b << "] ";
	return os;
}

