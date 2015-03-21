μPacket - A micro packet ray tracer
===
An extremely simple packet based ray tracer, uses AVX2 to trace eight rays at once through the scene. Currently only supports spheres and planes
with Lambertian BRDFs illuminated by a single point light. Illumination is computed with Whitted ray tracing, although recursion only goes as
far as computing shadows since there are no reflective or transmissive materials.

Building
---
The AVX2 instruction set is required, along with a relatively modern C++ compiler (some C++11/14 features are used), you should also compile an 64 bit executable.
The program should compile and run on at least VS 2013 Community and gcc 4.8.1+ on Linux. MinGW is not supported on Windows as it's unable to align the stack to 32 bytes,
see [bug](https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412). I haven't tested on clang but it might work. Beyond that only CMake is required to generate
build files for your build system.

Running
---
Running the executable built will produce the image below and save it to out.bmp.

![Render output](http://i.imgur.com/WcM6Rcl.png)

