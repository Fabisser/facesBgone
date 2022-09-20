cmake_minimum_required (VERSION 3.20)

project ("Geometry4CFD")

# find json.hpp
include_directories( ${CMAKE_SOURCE_DIR}/include/ ) # include json.hpp

# find CGAL
find_package(CGAL) # for CGAL
if (CGAL_FOUND)
	include(${CGAL_USE_FILE})
	message(STATUS "CGAL found")
else()
	message(SEND_ERROR "this code requires the CGAL library")
	return()
endif()

# define data path
add_definitions(
  -DDATA_PATH=\"${PROJECT_SOURCE_DIR}/data\"
)

# add cpp files
add_executable (Geometry4CFD "src/main.cpp")

