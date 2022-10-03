project(convex_hull)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_CXX_STANDARD 14)	

include_directories( ${CMAKE_SOURCE_DIR}/include ) # include json.hpp
include_directories( ${CMAKE_SOURCE_DIR}/src ) # include json.hpp

add_subdirectory(pybind11)

pybind11_add_module(convex_hull src/main.cpp)

find_package(CGAL) # for CGALgeocfd
if (CGAL_FOUND)
	include(${CGAL_USE_FILE})
	message(STATUS "CGAL found")
else()
	message(SEND_ERROR "this code requires the CGAL library")
	return()
endif()

target_link_libraries(convex_hull CGAL::CGAL)
include(CTest)
enable_testing()

# define data path
add_definitions(
  -DDATA_PATH=\"${PROJECT_SOURCE_DIR}/data\"
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_BUILD_TYPE Release)
include(CPack)
