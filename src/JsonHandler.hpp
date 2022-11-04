#pragma once

// include files
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>

#include "json.hpp"
#include <CGAL/Exact_predicates_exact_constructions_kernel.h> // Nef (and Minkowski) requires exact constructions

// typedefs
typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel; 
typedef Kernel::Point_3 Point_3;

// using declaration
using json = nlohmann::json;

// define constant epsilon - tolerance
const double epsilon = 1e-8;

// struct definitions - for reading from cityjson and store the information
struct Ring
{
	std::vector<unsigned long> indices;

	// ... may add other attributes
	// indicating the indices in "vertices"
	// e.g.
	// surface[[0,1,2,3], [4,5,6,7]] is a surface which has two rings:
	// [0,1,2,3] and [4,5,6,7]
};

struct Face
{
	std::vector<Ring> rings;

	// ... may add other attributes
	// e.g. 
	// [[0,1,2,3]] is a surface without any holes
	// [[0,1,2,3], [4,5,6,7]] is a surface(containing a hole) - has two rings
	// this surface has two rings: [0,1,2,3], [4,5,6,7]
	// for most surfaces, it only has one ring - e.g. surface [[1,2,6,5]] only has one ring: [1,2,6,5]
};

struct Shell
{
	std::vector<Face> faces;

	// ... may add other attributes
	// e.g.
	// [[[0,1,2,3]], [[3,4,5,6]], [[4,5,6,7]], [[7,8,9,10]]] - a shell containing 4 surfaces
};

struct Solid
{
	std::vector<Shell> shells; // store the shells
	std::string id; // store the building id, if needed
	double lod; // store the lod level, if needed - lod must be string(i.e. "1.3") when writing to file (otherwise invalid file)

	// ... may add other attributes
	// principally one solid can contain multiple shells
	// generally in cityjson file one solid contains one shell
	// e.g.
	// [
	// [[[0,1,2,3]], [[3,4,5,6]], [[4,5,6,7]], [[7,8,9,10]]], -- shell 1
	// [[[0,2,1,3]], [[6,5,4,3]], [[7,6,5,4]], [[10,9,8,7]]], -- shell 2
	// ]
};
// end of struct definitions - for reading from cityjson and store the information



// handle cityjson file
class JsonHandler
{
protected:
	/*
	* check if a vertex already exists in a vertices vector
	* it's important to convert Point_3's coordinates(x, y, z) to double first
	* since we are using Exact_predicates_exact_constructions_kernel
	* use double coordinates to compare whether two vertices are the same
	* return: False - not exist, True - already exist
	*/
	bool vertex_exist_check(std::vector<Point_3>& vertices, const Point_3& vertex) {
		bool flag(false);
		for (const auto& v : vertices) {
			if (
				abs(CGAL::to_double(vertex.x()) - CGAL::to_double(v.x())) < epsilon &&
				abs(CGAL::to_double(vertex.y()) - CGAL::to_double(v.y())) < epsilon &&
				abs(CGAL::to_double(vertex.z()) - CGAL::to_double(v.z())) < epsilon) {
				flag = true;
			}
		}
		return flag;
	}


	/*
	* if a vertex is repeated, find the index in vertices vector and return the index
	* it's important to convert Point_3's coordinates(x, y, z) to double first
	* since we are using Exact_predicates_exact_constructions_kernel
	*/
	unsigned long find_vertex_index(std::vector<Point_3>& vertices, Point_3& vertex) {
		for (std::size_t i = 0; i != vertices.size(); ++i) {
			if (
				abs(CGAL::to_double(vertex.x()) - CGAL::to_double(vertices[i].x())) < epsilon &&
				abs(CGAL::to_double(vertex.y()) - CGAL::to_double(vertices[i].y())) < epsilon &&
				abs(CGAL::to_double(vertex.z()) - CGAL::to_double(vertices[i].z())) < epsilon) {
				return (unsigned long)i;
			}
		}
		std::cout << "warning: please check find_vertex_index function, no index found" << '\n';
		return 0;
	}

public:



	/*
	* shift the coordinates use bounding box
	* calculate xmin, ymin, zmin
	* for each vertex in json, do:
	* x = x - xmin;
	* y = y - ymin;
	* z = z - zmin;
	* 
	* this function will be called for the whole json tile
	*/
	static std::tuple<double, double, double> get_translation_datum(json& j, double lod) {

		double xmin = 1e12;
		double ymin = 1e12;
		double zmin = 1e12;

		int count = 0;

		for (auto& co : j["CityObjects"].items()) {
			for (auto& g : co.value()["geometry"]) {
				if (g["type"] == "Solid" && (std::abs(g["lod"].get<double>() - lod)) < epsilon) { // geometry type: Solid
					for (auto& shell : g["boundaries"]) {
						for (auto& surface : shell) {
							for (auto& ring : surface) {
								for (auto& v : ring)
								{
									std::vector<int> vi = j["vertices"][v.get<int>()];
									double x = (vi[0] * j["transform"]["scale"][0].get<double>()) + j["transform"]["translate"][0].get<double>();
									double y = (vi[1] * j["transform"]["scale"][1].get<double>()) + j["transform"]["translate"][1].get<double>();
									double z = (vi[2] * j["transform"]["scale"][2].get<double>()) + j["transform"]["translate"][2].get<double>();

									if ((x - xmin) < epsilon)
										xmin = x;
									if ((y - ymin) < epsilon)
										ymin = y;
									if ((z - zmin) < epsilon)
										zmin = z;

								} // end for: each indice in one ring	
							}// end for: each ring in one surface	
						} // end for: each surface in one shell
					}// end for: each shell in one solid
					++count;
				}// end if: solid 
			}
			
		}

		std::cout << "buildings count in the input json file: " << count << '\n';
		std::cout << "xmin: " << xmin << '\n';
		std::cout << "ymin: " << ymin << '\n';
		std::cout << "zmin: " << zmin << '\n';

		return std::make_tuple(xmin, ymin, zmin);
		
	}



	/*
	* CityJSON files have their vertices compressed : https://www.cityjson.org/specs/1.1.1/#transform-object
	* lod specified: 1.2 & 1.3 & 2.2
	* datum: contains xmin, ymin, zmin for shifting coordinates
	*/
	void read_certain_building(
		const json& j, 
		const std::string& building_id, 
		double lod,
		std::tuple<double, double, double>& datum) 
	{
		for (auto& co : j["CityObjects"].items()) {
			if (co.key() == building_id)
			{
				//std::cout << co.key() << '\n';
				for (auto& g : co.value()["geometry"]) {
					if (g["type"] == "Solid" && (std::abs(g["lod"].get<double>() - lod)) < epsilon) { // geometry type: Solid
						//std::cout << "lod level: " << g["lod"].get<double>() << '\n';
						Solid so; // create a solid to store the information
						so.id = co.key(); // store id
						so.lod = g["lod"].get<double>(); // store lod info as std::string type
						for (auto& shell : g["boundaries"]) {
							Shell se;
							for (auto& surface : shell) {
								Face f; // create a face
								for (auto& ring : surface) {
									Ring r; // create a ring
									//std::cout << "---" << std::endl;
									for (auto& v : ring)
									{
										std::vector<int> vi = j["vertices"][v.get<int>()];
										double x = (vi[0] * j["transform"]["scale"][0].get<double>()) + j["transform"]["translate"][0].get<double>();
										double y = (vi[1] * j["transform"]["scale"][1].get<double>()) + j["transform"]["translate"][1].get<double>();
										double z = (vi[2] * j["transform"]["scale"][2].get<double>()) + j["transform"]["translate"][2].get<double>();

										// get the translation datum
										double xmin = std::get<0>(datum);
										double ymin = std::get<1>(datum);
										double zmin = std::get<2>(datum);
										
										// shift the coordinates
										x = x - xmin;
										y = y - ymin;
										z = z - zmin;

										// when adding new vertex and adding new index in r.indices, check repeatness
										Point_3 new_vertex(x, y, z);
										bool if_existed = vertex_exist_check(vertices, new_vertex);
										if (!if_existed) {
											vertices.emplace_back();
											vertices.back() = new_vertex; // if not existed yet, add it to vertices vector
											r.indices.emplace_back((unsigned long)vertices.size() - 1); // add new index to this ring's indices vector
											// since we add new vertex to vertices, vertices.size()-1 represents the last index
											// can also do: r.indices.emplace_back(index++);
										}
										else {
											unsigned long exist_index = find_vertex_index(vertices, new_vertex);
											r.indices.emplace_back();
											r.indices.back() = exist_index; // if existed, add the exist index to this ring's indices vector
										}

										// vertices.emplace_back();
										// vertices.back() = Point_3(x, y, z); // add the vertex to the vertices vector
										// std::cout << v << " (" << x << ", " << y << ", " << z << ")" << '\n';
										// r.indices.emplace_back((unsigned long)vertices.size()-1); // new indices 0-based

									} // end for: each indice in one ring
									f.rings.emplace_back(r); // add ring to the surface
								}// end for: each ring in one surface
								se.faces.emplace_back(f);
							} // end for: each surface in one shell
							so.shells.emplace_back(se);
						}// end for: each shell in one solid
						solids.emplace_back(so);
					}// end if: solid 
				}
			}
		}
	}



	/*
	* prompt basic information of the current building
	*/
	void message()
	{		
		std::cout << "building(part) name: " << solids[0].id << '\n';
		std::cout << "lod level: " << solids[0].lod << '\n';
		std::cout << "number of vertices: " << vertices.size() << '\n';
		std::cout << '\n';		
	}


protected:
	std::vector<Point_3> vertices; // store all vertices of one building
	std::vector<Solid> solids; // store all solids of one building, ideally one solid for each building

	friend class Build; // friend class to access the protected members
};



// read from files
namespace FileIO {

	// declaration for convenient use - whenever use specify the scope
	using std::vector;
	using std::string;

	/*
	* read the adjacency txt file to get the adjacency buildings
	*/
	void read_adjacency_from_txt(const std::string& filename, vector<string>& adjacency) {

		// read from txt to get the adjacency list
		std::ifstream in(filename, std::ios::out);
		if (!in.is_open()) {
			std::cerr << "Error: Unable to open adjacency file \"" << filename << "\" for reading!" << std::endl;
			return;
		}

		std::string line;
		while (std::getline(in, line)) {
			adjacency.emplace_back(line);
		}
		in.close();

	}



	/*
	* read the all_adjacency txt file to get all adjacent buildings
	*/
	void read_all_adjacencies_from_txt(const std::string& filename, vector<vector<string>>& adjacencies) {
		// read from txt to get the adjacency list
		std::ifstream in(filename, std::ios::out);
		if (!in.is_open()) {
			std::cerr << "Error: Unable to open adjacency file \"" << filename << "\" for reading!" << std::endl;
			return;
		}

		// read line by line, add building names to adjacency
		// an empty line indicates another adjacency
		vector<string> adjacency;
		std::string line;
		while (std::getline(in, line)) {
			if (line != "") {
				adjacency.emplace_back(line);
			}else {
				adjacencies.emplace_back(adjacency);
				adjacency.clear();
			}
		}
		in.close();

		// prompt info
		std::cout << "adjacencies size: " << adjacencies.size() << '\n';
		/*for (const auto& adjacency : adjacencies) {
			for (const auto& name : adjacency)std::cout << name << '\n';
			std::cout << '\n';
		}*/
	}
}

