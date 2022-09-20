cmake_minimum_required(VERSION 3.0.0)
project(geocfd VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11) # cpp17 does not understand standard libraries

include_directories( ${CMAKE_SOURCE_DIR}/include ) # include json.hpp

find_package(CGAL) # for CGALgeocfd
if (CGAL_FOUND)
	include(${CGAL_USE_FILE})
	message(STATUS "CGAL found")
else()
	message(SEND_ERROR "this code requires the CGAL library")
	return()
endif()

include(CTest)
enable_testing()

# define data path
add_definitions(
  -DDATA_PATH=\"${PROJECT_SOURCE_DIR}/data\"
)

add_executable(geocfd src/main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
