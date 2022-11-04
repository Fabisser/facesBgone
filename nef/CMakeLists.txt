cmake_minimum_required (VERSION 3.8)

project ("geoCFD")

set (CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_SOURCE_DIR}/include) # include json.hpp

#set(${CGAL_DIR} /mnt/c/Users/Panagiotior/Desktop/Test_Jupyter/Synthesis-Project/PolyFit-CGAL/thirdparty/CGAL-5.5.1)
#set(${CGAL_USE_FILE} /mnt/c/Users/Panagiotior/Desktop/Test_Jupyter/Synthesis-Project/PolyFit-CGAL/thirdparty/CGAL-5.5.1/cmake/modules/UseCGAL.cmake)
#message(STATUS ${CGAL_USE_FILE})
#include(${CGAL_USE_FILE})

find_package(CGAL) # for CGALgeocfd
if (CGAL_FOUND)
	include(${CGAL_USE_FILE})
	message(STATUS "CGAL found")
	message(STATUS "CGAL VERSION" " " ${CGAL_VERSION})
	message(STATUS ${CGAL_DIR})
	message(STATUS ${CGAL_USE_FILE})
else()
	message(SEND_ERROR "this code requires the CGAL library")
	return()
endif()

add_executable (geoCFD "src/main.cpp" "src/JsonHandler.hpp" "src/Polyhedron.hpp" "src/JsonWriter.hpp"  "src/MultiThread.hpp")

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
