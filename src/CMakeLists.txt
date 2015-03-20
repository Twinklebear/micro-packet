add_executable(micro_packet main.cpp vec.cpp color.cpp render_target.cpp)

install(TARGETS micro_packet DESTINATION ${MICRO_PACKET_INSTALL_DIR})

