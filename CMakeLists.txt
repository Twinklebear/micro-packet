cmake_minimum_required(VERSION 3.1)
project(MICRO_PACKET)

if (${WIN32})
	message(FATAL_ERROR "Windows is not supported")
endif()

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -march=native ${OpenMP_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -ggdb -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -DNDEBUG -fno-exceptions")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} -O3 -g -DNDEBUG -fno-exceptions")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fno-exceptions ${OpenMP_EXE_LINKER_FLAGS}")

# Detect ISA to pick DEFAULT_WIDTH for psimd's packs
include(cmake/check_isa_default.cmake)
check_isa_default(MICRO_PACKET_ISA_DEFAULT)
if (${MICRO_PACKET_ISA_DEFAULT} STREQUAL "SSE2" OR ${MICRO_PACKET_ISA_DEFAULT} STREQUAL "SSE4.2")
	message("Vector ISA is 4-wide")
	add_definitions(-DDEFAULT_WIDTH=4)
elseif (${MICRO_PACKET_ISA_DEFAULT} STREQUAL "AVX" OR ${MICRO_PACKET_ISA_DEFAULT} STREQUAL "AVX2")
	message("Vector ISA is 8-wide")
	add_definitions(-DDEFAULT_WIDTH=8)
elseif (${MICRO_PACKET_ISA_DEFAULT} STREQUAL "AVX512KNL" OR ${MICRO_PACKET_ISA_DEFAULT} STREQUAL "AVX512SKX")
	message("Vector ISA is 16-wide")
	add_definitions(-DDEFAULT_WIDTH=16)
else ()
	message(WARNING "Could not detect your vector ISA, using default pack width")
endif()

include_directories(include psimd)
add_subdirectory(src)

