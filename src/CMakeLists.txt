add_executable(micro_packet main.cpp vec.cpp color.cpp render_target.cpp camera.cpp sphere.cpp
	plane.cpp light.cpp scene.cpp block_queue.cpp ld_sampler.cpp)

install(TARGETS micro_packet DESTINATION ${MICRO_PACKET_INSTALL_DIR})

