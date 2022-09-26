#include "Polyhedron.h"
#include <CGAL/Polyhedron_3.h>

#define DATA_PATH "/Users/fabzv/Desktop/Delft/Synthesis-Project/data"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


int calculate(std::vector<std::vector<std::string>> buildings)
{

    // std::cout << "-- activated data folder: " << DATA_PATH << '\n';

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

  //read certain building

  for (auto adjbuildings : buildings) {
    for (auto b : adjbuildings) {
      JsonHandler jhandle;
      jhandle.read_certain_building(j, b);
      jhandle.message();
      BuildPolyhedron::build_one_polyhedron(jhandle);
    }
  }
  return 0;
}

PYBIND11_MODULE(convex_hull, module_handle) {
  module_handle.doc() = "Create Nef Polyhedra";
  module_handle.def("calculate", &calculate);
}
