#pragma once

// include files
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include "json.hpp"


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
	double lod; // store the lod level, if needed - lod must be converted to string(i.e. "1.3") when writing to file

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
	* use coordinates to compare whether two vertices are the same
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
	* CityJSON files have their vertices compressed : https://www.cityjson.org/specs/1.1.1/#transform-object
	* this function visits all the surfaces of a certain building 
	* and print the (x,y,z) coordinates of each vertex encountered
	* lod specified: 1.3
	*/
	void read_certain_building(const json& j, const std::string& building_id) {
		for (auto& co : j["CityObjects"].items()) {
			if (co.key() == building_id)
			{
				std::cout << "CityObject: " << co.key() << std::endl;
				for (auto& g : co.value()["geometry"]) {
					if (g["type"] == "Solid" && (abs(g["lod"].get<double>() - 1.3)) < epsilon) { // geometry type: Solid, use lod1.3
						std::cout << "current lod level: " << g["lod"].get<double>() << '\n';
						Solid so; // create a solid to store the information
						so.id = co.key(); // store id
						so.lod = g["lod"].get<double>(); // store lod info
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
										
										// when adding new vertex and adding new index in r.indices, check repeatness
										Point_3 new_vertex(x, y, z);
										bool if_existed = vertex_exist_check(vertices, new_vertex);
										if(!if_existed){
											vertices.emplace_back();
											vertices.back() = new_vertex; // if not existed yet, add it to vertices vector
											r.indices.emplace_back((unsigned long)vertices.size()-1); // add new index to this ring's indices vector
											// since we add new vertex to vertices, vertices.size()-1 represents the last index
											// can also do: r.indices.emplace_back(index++);
										}
										else{
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
					}// end if: solid and lod = 1.3
				}
			}
		}
	}


	
	/*
	* prompt basic information of the current building
	*/
	void message()
	{
		std::cout<<"---------building(part) info----------\n";
		std::cout<<"building(part) name: "<<solids[0].id<<'\n';
		std::cout<<"lod level: "<<solids[0].lod<<'\n';
		std::cout<<"number of vertices: "<<vertices.size()<<'\n';
		std::cout<<"--------------------------------------\n";
	}
	
	
	/*
	* write the selected building to cityjson file
	* index: index of solids, indicating which solid is going to be written to the json file
	* this needs to be altered to write the big nef to cityjson
	*/
	// void write_json_file(const std::string& filename, const std::size_t index)
	// {
	// 	// basic info ---------------------------------------------------------------
	// 	json js;
	// 	js["type"] = "CityJSON";
	// 	js["version"] = "1.1";
	// 	js["transform"] = json::object();
	// 	js["transform"]["scale"] = json::array({ 1.0, 1.0, 1.0 });
	// 	js["transform"]["translate"] = json::array({ 0.0, 0.0, 0.0 });
	// 	js["vertices"] = json::array({}); // vertices

	// 	// all vertices(including repeated vertices)-----------------------------------		
	// 	for (auto const& v : vertices) {
	// 		double x = CGAL::to_double(v.x()); // warning: may have precision loss
	// 		double y = CGAL::to_double(v.y());
	// 		double z = CGAL::to_double(v.z());
	// 		js["vertices"].push_back({ x, y, z });
	// 	}

	// 	// names
	// 	std::string bp_name = solids[index].id; // BuildingPart's name, i.e. "NL.IMBAG.Pand.0503100000019695-0"
	// 	std::string b_name = bp_name.substr(0, bp_name.length() - 2); // Building's name, i.e. "NL.IMBAG.Pand.0503100000019695"

	// 	// Building info -------------------------------------------------------------------
	// 	js["CityObjects"] = json::object();
	// 	js["CityObjects"][b_name]["type"] = "Building";
	// 	js["CityObjects"][b_name]["attributes"] = json({});
	// 	js["CityObjects"][b_name]["children"] = json::array({ bp_name });
	// 	js["CityObjects"][b_name]["geometry"] = json::array({});

	// 	// BuildingPart info ---------------------------------------------------------------
	// 	js["CityObjects"][bp_name]["type"] = "BuildingPart";
	// 	js["CityObjects"][bp_name]["attributes"] = json({});
	// 	js["CityObjects"][bp_name]["parents"] = json::array({ b_name });

	// 	// geometry
	// 	js["CityObjects"][bp_name]["geometry"] = json::array();
	// 	js["CityObjects"][bp_name]["geometry"][0]["type"] = "Solid";
	// 	js["CityObjects"][bp_name]["geometry"][0]["lod"] = "1.3"; // lod must be string, otherwise invalid file
	// 	js["CityObjects"][bp_name]["geometry"][0]["boundaries"] = json::array({}); // indices	

	// 	// boundaries
	// 	auto& boundaries = js["CityObjects"][bp_name]["geometry"][0]["boundaries"][0];
	// 	const auto& so = solids[index];
	// 	for (const auto& se : so.shells)
	// 		for (const auto& f : se.faces)
	// 			for(const auto& r : f.rings)
	// 				boundaries.push_back({ r.indices });
	
	// 	// write to file
	// 	std::string json_string = js.dump(2);
	// 	std::ofstream out_stream(filename);
	// 	out_stream << json_string;
	// 	out_stream.close();
	// 	std::cout << "file saved at: " << filename << '\n';
	// }


public:
	std::vector<Point_3> vertices; // store all vertices of one building
	std::vector<Solid> solids; // store all solids of one building, ideally one solid for each building
};
