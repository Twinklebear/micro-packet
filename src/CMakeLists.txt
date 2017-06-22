add_executable(micro_packet main.cpp vec.cpp color.cpp render_target.cpp camera.cpp sphere.cpp
	plane.cpp light.cpp scene.cpp block_queue.cpp ld_sampler.cpp)
set_property(TARGET micro_packet PROPERTY CXX_STANDARD 14)

