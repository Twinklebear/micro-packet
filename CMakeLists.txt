cmake_minimum_required(VERSION 2.6)
project(MICRO_PACKET)

# Bump up warning levels appropriately for each compiler
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pedantic -std=c++14 -m64 -march=native")
	set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -ggdb -DDEBUG")
	set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -flto -fno-use-linker-plugin -DNDEBUG -fno-exceptions")
	set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -flto -fno-exceptions")
	if (${WIN32})
		message(WARNING "MinGW can't align the stack to 32-bytes on Windows, program will probably crash")
	endif()
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Zi")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4 /Zi")
	endif()
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX /arch:AVX2")
endif()

set(MICRO_PACKET_INSTALL_DIR "${MICRO_PACKET_SOURCE_DIR}/bin")

add_subdirectory(src)

