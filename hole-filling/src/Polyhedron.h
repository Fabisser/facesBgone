#pragma once

// JsonHandler
#include "JsonHandler.h"

// necessary include files from CGAL
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/minkowski_sum_3.h>

// typedefs
typedef CGAL::Polyhedron_3<Kernel> Polyhedron;
typedef CGAL::Nef_polyhedron_3<Kernel> Nef_polyhedron;


/*
load each building(or solid) into a CGAL Polyhedron_3 using the Polyhedron_incremental_builder_3. 
In order to use the Polyhedron_incremental_builder_3, you need to create a custom struct or class.
*/
template <class HDS>
struct Polyhedron_builder : public CGAL::Modifier_base<HDS> {
    std::vector<Point_3> vertices; // type: Kernel::Point_3
    std::vector<std::vector<unsigned long>> faces; // INDEX for vertices

    Polyhedron_builder() {}
    void operator()(HDS& hds) {
        CGAL::Polyhedron_incremental_builder_3<HDS> builder(hds, true);
        std::cout << "building surface with " << vertices.size() << " vertices and " << faces.size() << " faces" << '\n';

        builder.begin_surface(vertices.size(), faces.size());
        for (auto const& vertex : vertices) builder.add_vertex(vertex);
        for (auto const& face : faces) builder.add_facet(face.begin(), face.end());
        builder.end_surface();
    }
};


/*
use CGAL to build polyhedron
*/
class BuildPolyhedron
{
public:

    // build one polyhedron using vertices and faces from one shell (one building)
    // jhandle: A JsonHandler instance, contains all vertices and solids
    // index  : index of solids vector, indicating which solid is going to be built - ideally one building just contains one solid
    static void build_one_polyhedron(const JsonHandler& jhandle, unsigned long index=0)
    {
        const auto& solid = jhandle.solids[index]; // get the solid
        if(solid.shells.size() != 1){
            std::cout<<"warning: this solid contains 0 or more than one shells\n";
            std::cout<<"please check build_one_polyhedron function and check the following solid:\n";
            std::cout<<"solid id: "<<solid.id<<'\n';
            std::cout<<"solid lod: "<<solid.lod<<'\n';
            std::cout<<"no polyhedron is built with this solid\n";
        }
        else{
            // create a polyhedron and a builder
            Polyhedron polyhedron;
            Polyhedron_builder<Polyhedron::HalfedgeDS> polyhedron_builder;

            // add vertices and faces to polyhedron_builder
            polyhedron_builder.vertices = jhandle.vertices; // now jhandle only handles one building(solid)
            for(auto const& shell : solid.shells)
                for(auto const& face : shell.faces)
                    for(auto const& ring : face.rings)
                        polyhedron_builder.faces.push_back(ring.indices);
            
            // call the delegate function
            polyhedron.delegate(polyhedron_builder);
            std::cout<<"polyhedron closed? "<<polyhedron.is_closed()<<'\n';
            if(polyhedron.is_closed()){
                Nef_polyhedron nef_polyhedron(polyhedron);
                std::cout<<"build nef polyhedron"<<'\n';
            }
        }
       
    }
};