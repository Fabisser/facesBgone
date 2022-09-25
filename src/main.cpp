#include <gmp.h>
#include <mpfr.h>
#include "Polyhedron.h"
#include <CGAL/Polyhedron_3.h>
#define DATA_PATH "/home/fabisser/Synthesis-Project-Fengyan/data"

#include <pybind11/pybind11.h>

namespace py = pybind11;

int calculate(py::str buildings)
{

  std::cout << "-- activated data folder: " << DATA_PATH << '\n';

  //  //  std::cout<<"newly-added\n";
  //  //std::cout<<"data path is: "<<mypath<<'\n';
    
  //  //  char buffer[256];
  //  //  if (getcwd(buffer, sizeof(buffer)) != NULL) {
  //  //     printf("Current working directory : %s\n", buffer);
  //  //  } else {
  //  //     perror("getcwd() error");
  //  //     return 1;
  //  //  }

   //-- reading the (original)file with nlohmann json: https://github.com/nlohmann/json  
   std::string filename = "/3dbag_v210908_fd2cee53_5907.json";
   std::cout << "current reading file is: " << DATA_PATH + filename << '\n';
   std::ifstream input(DATA_PATH + filename);
   json j;
   input >> j;
   input.close();

  //  //read certain building

  // //  for (auto adjbuildings : buildings) {
  // //   for (auto b : adjbuildings) {
  JsonHandler jhandle;
  // std::string build = std::string(buildings);
  // std::cout << "created nef of" << build << std::endl;
  jhandle.read_certain_building(j, "test");
  // jhandle.message();
  // BuildPolyhedron::build_one_polyhedron(jhandle);
      
  //   }
  //  }

   // // test output
   //
   // std::cout << "vertices number: " << '\n';
   // for (const auto& so : jhandle.solids)
   //    for (const auto& se : so.shells)
   //       for (const auto& f : se.faces)
   //          for (const auto& r : f.rings) // for most cases, each face only contains one ring -> i.e. face [[0,1,2,3]] only has one ring
   //             {
   //                std::cout << "--------" << '\n';
   //                for (const auto& indice : r.indices)
   //                   std::cout << indice << '\n';
   //             }

   // write file
   //std::string writeFilename = "/SimpleBuildings.json";
   //jhandle.write_json_file(DATA_PATH + writeFilename, 0); // second argument: indicating which solid is going to be written to the file
                    

	return 0;
}

PYBIND11_MODULE(convex_hull, module_handle) {
  module_handle.doc() = "Create Nef Polyhedra";
  module_handle.def("calculate", &calculate);
}
