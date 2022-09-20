#include <iostream>
#include "json.hpp"

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/minkowski_sum_3.h>

typedef CGAL::Exact_predicates_exact_constructions_kernel Kernel;
typedef Kernel::Point_3 Point;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron;

using json = nlohmann::json;

const double epsilon = 1e-8; // the tolerance

// CityJSON files have their vertices compressed: https://www.cityjson.org/specs/1.1.1/#transform-object
// this function visits all the surfaces of a certain building 
// and print the (x,y,z) coordinates of each vertex encountered
void read_certain_building(const json& j, const std::string& building_id) {
    for (auto& co : j["CityObjects"].items()) {
        if (co.key() == building_id)
        {
            std::cout << "CityObject: " << co.key() << std::endl;
            for (auto& g : co.value()["geometry"]) {
                if (g["type"] == "Solid" && (abs(g["lod"].get<double>()-1.3)) < epsilon) { // geometry type: Solid, use lod1.3
                    std::cout <<"current lod level: " << g["lod"].get<double>() << '\n';
                    for (auto& shell : g["boundaries"]) {
                        for (auto& surface : shell) {
                            for (auto& ring : surface) {
                                std::cout << "---" << std::endl;
                                for (auto& v : ring)
                                {
                                    std::vector<int> vi = j["vertices"][v.get<int>()];
                                    double x = (vi[0] * j["transform"]["scale"][0].get<double>()) + j["transform"]["translate"][0].get<double>();
                                    double y = (vi[1] * j["transform"]["scale"][1].get<double>()) + j["transform"]["translate"][1].get<double>();
                                    double z = (vi[2] * j["transform"]["scale"][2].get<double>()) + j["transform"]["translate"][2].get<double>();
                                    //std::cout << std::setprecision(2) << std::fixed << v << " (" << x << ", " << y << ", " << z << ")" << std::endl;
                                    std::cout << v << " (" << x << ", " << y << ", " << z << ")" << '\n';
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


int main()
{
	std::cout << "Hello CMake." << std::endl;

    //-- reading the (original)file with nlohmann json: https://github.com/nlohmann/json  
    std::string filename = "/3dbag_v210908_fd2cee53_5907.json";
    std::cout << "current reading file is: " << DATA_PATH + filename << '\n';
    std::ifstream input(DATA_PATH + filename);
    json j;
    input >> j;
    input.close();

    // read certain building
    std::string building_id = "NL.IMBAG.Pand.0503100000019695-0";
    read_certain_building(j, building_id);
	return 0;
}
