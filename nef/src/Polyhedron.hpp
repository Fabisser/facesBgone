#pragma once

// JsonHandler
#include "JsonHandler.hpp"

// necessary include files from CGAL
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Nef_polyhedron_3.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/minkowski_sum_3.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h> // for filling holes
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h> // for triangulating surfaces
//#include <CGAL/Polygon_mesh_processing/triangulate_hole.h> // for filling holes
#include <boost/foreach.hpp> // for filling holes
#include <CGAL/OFF_to_nef_3.h> // for erosion - constructing bbox


// for remeshing
#include <CGAL/Surface_mesh.h> // for surface_mesh
#include <CGAL/boost/graph/copy_face_graph.h> // for converting to surface mesh
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/boost/graph/convert_nef_polyhedron_to_polygon_mesh.h>
#include <boost/iterator/function_output_iterator.hpp>


// typedefs
typedef CGAL::Polyhedron_3<Kernel>                   Polyhedron;
typedef CGAL::Nef_polyhedron_3<Kernel>               Nef_polyhedron;
typedef Polyhedron::Facet_iterator                   Facet_iterator; // for extracting geometries
typedef Polyhedron::Halfedge_around_facet_circulator Halfedge_facet_circulator; // for extracting geometries
typedef Polyhedron::Halfedge_handle                  Halfedge_handle; // for filling holes
typedef Polyhedron::Facet_handle                     Facet_handle; // for filling holes
typedef Polyhedron::Vertex_handle                    Vertex_handle; // for filling holes

// for remeshing
typedef CGAL::Surface_mesh<Kernel::Point_3>                   Mesh;
typedef boost::graph_traits<Mesh>::halfedge_descriptor        halfedge_descriptor;
typedef boost::graph_traits<Mesh>::edge_descriptor            edge_descriptor;



namespace PMP = CGAL::Polygon_mesh_processing;



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
        //std::cout << "building surface with " << vertices.size() << " vertices and " << faces.size() << " faces" << '\n';

        builder.begin_surface(vertices.size(), faces.size());
        for (auto const& vertex : vertices) builder.add_vertex(vertex);
        for (auto const& face : faces) builder.add_facet(face.begin(), face.end());
        builder.end_surface();
    }
};


/*
use CGAL to build polyhedron
*/
class Build
{
public:

    // build one polyhedron using vertices and faces from one shell (one building)
    // jhandle: A JsonHandler instance, contains all vertices and solids
    // index  : index of solids vector, indicating which solid is going to be built - ideally one building just contains one solid
    // triangulate: if true, triangulation of surfaces will be performed before building nef
    static void build_nef_polyhedron(
        const JsonHandler& jhandle, 
        std::vector<Nef_polyhedron>& Nefs,
        bool triangulate = true,
        unsigned long index = 0)
    {
        const auto& solid = jhandle.solids[index]; // get the solid

        //std::cout << solid.id << '\n';

        if (solid.shells.size() != 1) {
            std::cout << "warning: this solid contains 0 or more than one shells\n";
            std::cout << "please check build_one_polyhedron function and check the following solid:\n";
            std::cout << "solid id: " << solid.id << '\n';
            std::cout << "solid lod: " << solid.lod << '\n';
            std::cout << "no polyhedron is built with this solid\n";
        }
        else {
            // create a polyhedron and a builder
            Polyhedron polyhedron;
            Polyhedron_builder<Polyhedron::HalfedgeDS> polyhedron_builder;

            // add vertices and faces to polyhedron_builder
            polyhedron_builder.vertices = jhandle.vertices; // now jhandle only handles one building(solid)
            for (auto const& shell : solid.shells)
                for (auto const& face : shell.faces)
                    for (auto const& ring : face.rings)
                        polyhedron_builder.faces.push_back(ring.indices);

            // call the delegate function
            polyhedron.delegate(polyhedron_builder);
            //std::cout << "polyhedron closed? " << polyhedron.is_closed() << '\n';

            if (polyhedron.is_closed()) {

                // filling holes?
                // polyhedron_hole_filling(polyhedron);

                // if triangulation is true, triangulate the surfaces first (lod2.2)
                if (triangulate) {
                    CGAL::Polygon_mesh_processing::triangulate_faces(polyhedron);
                }

                // build nef polyhedron
                Nef_polyhedron nef_polyhedron(polyhedron);
                Nefs.emplace_back();
                Nefs.back() = nef_polyhedron; // add the built nef_polyhedron to the Nefs vector
                std::cout << "build nef polyhedron" << " ";
                std::cout << "-> " << (nef_polyhedron.is_valid() ? "valid" : "invalid") << '\n';
            }
            else {
                std::cout << "the polyhedron is not closed, build convex hull to replace it" << '\n';
                std::cout << "building id: " << solid.id << '\n';
                Polyhedron convex_polyhedron;
                CGAL::convex_hull_3(jhandle.vertices.begin(), jhandle.vertices.end(), convex_polyhedron);

                // now check if we successfully build the convex hull
                if (convex_polyhedron.is_closed()) {

                    // if triangulation is true, triangulate the surfaces first (lod2.2)
                    if (triangulate) {
                        CGAL::Polygon_mesh_processing::triangulate_faces(convex_polyhedron);
                    }

                    // get nef polyhedron of the convex hull
                    Nef_polyhedron convex_nef_polyhedron(convex_polyhedron);
                    Nefs.emplace_back();
                    Nefs.back() = convex_nef_polyhedron;
                    std::cout<< "the convex hull is closed, build convex nef polyhedron" << '\n';
                }
                else {
                    std::cerr << "convex hull is not closed, no nef polyhedron built\n";
                }
            }

            /* test to write the polyhedron to .off file --------------------------------------------------------*/

            // Write polyhedron in Object File Format (OFF).
            // CGAL::set_ascii_mode(std::cout);
            // std::cout << "OFF" << std::endl << polyhedron.size_of_vertices() << ' '
            //     << polyhedron.size_of_facets() << " 0" << std::endl;
            // std::copy(polyhedron.points_begin(), polyhedron.points_end(),
            //     std::ostream_iterator<Point_3>( std::cout, "\n"));
            // for (Facet_iterator i = polyhedron.facets_begin(); i != polyhedron.facets_end(); ++i) 
            // {
            //     Halfedge_facet_circulator j = i->facet_begin();
            //     // Facets in polyhedral surfaces are at least triangles.
            //     std::cout << CGAL::circulator_size(j) << ' ';
            //     do {
            //         std::cout << ' ' << std::distance(polyhedron.vertices_begin(), j->vertex());
            //     } while ( ++j != i->facet_begin());
            //     std::cout << std::endl;
            // }

            // std::cout<<polyhedron<<std::endl;


        }

    }


    /*
    * test hole filling package
    * not working for holes in dataset_2
    * building id: NL.IMBAG.Pand.0503100000029345-0
    * maybe because the holes are not reachable from the border?
    * since for each Halfedge_handle h h->is_border() is false
    */
    static void polyhedron_hole_filling(Polyhedron& poly) {

        // output it
        std::ofstream out("unfilled.off");
        out.precision(17);
        out << poly << std::endl;

        // Incrementally fill the holes
        unsigned int nb_holes = 0;
        for (Halfedge_handle h : halfedges(poly))
        {
            if (h->is_border())
            {
                std::vector<Facet_handle>  patch_facets;
                std::vector<Vertex_handle> patch_vertices;
                PMP::triangulate_and_refine_hole(poly,
                    h,
                    std::back_inserter(patch_facets),
                    std::back_inserter(patch_vertices),
                    CGAL::parameters::vertex_point_map(get(CGAL::vertex_point, poly))
                    .geom_traits(Kernel()));
                std::cout << " Number of facets in constructed patch: " << patch_facets.size() << std::endl;
                std::cout << " Number of vertices in constructed patch: " << patch_vertices.size() << std::endl;
                ++nb_holes;
            }
        }
        std::cout << std::endl;
        std::cout << nb_holes << " holes have been filled" << std::endl;
        std::ofstream out_filled("filled.off");
        out_filled.precision(17);
        out_filled << poly << std::endl;
        
        // Incrementally fill the holes
        // using border is not suitable in this case
        // since the holes are not reachable from the border?

        /*std::cout << "filling holes ..." << '\n';
        unsigned int nb_holes = 0;
        BOOST_FOREACH(Halfedge_handle h, halfedges(poly))
        {
            if (h->is_border())
            {
                std::vector<Facet_handle>  patch_facets;
                std::vector<Vertex_handle> patch_vertices;
                bool success = CGAL::cpp11::get<0>(
                    CGAL::Polygon_mesh_processing::triangulate_refine_and_fair_hole(
                        poly,
                        h,
                        std::back_inserter(patch_facets),
                        std::back_inserter(patch_vertices),
                        CGAL::Polygon_mesh_processing::parameters::vertex_point_map(get(CGAL::vertex_point, poly)).
                        geom_traits(Kernel())));
                std::cout << " Number of facets in constructed patch: " << patch_facets.size() << std::endl;
                std::cout << " Number of vertices in constructed patch: " << patch_vertices.size() << std::endl;
                std::cout << " Fairing : " << (success ? "succeeded" : "failed") << std::endl;
                ++nb_holes;
            }
        }
        std::cout << std::endl;
        std::cout << nb_holes << " holes have been filled" << std::endl;*/
    }

    
};



/*
* ------------------------------------------------------------------------------------------------------------------------------------------
* now we have finished building nef polyhedron, we need to further process to extract
* the geometries and prepare for writing to json file.
* ------------------------------------------------------------------------------------------------------------------------------------------
*/



// extract geometries
// guidance: https://3d.bk.tudelft.nl/courses/geo1004//hw/3/#4-extracting-geometries
// credit: Ken Ohori
struct Shell_explorer {
    std::vector<Point_3> vertices; // store the vertices for one shell when extracting geometries from a Nef polyhedron
    std::vector<std::vector<unsigned long>> faces; // store faces for one shell when extracting geometries from a  Nef polyhedron

    std::vector<Point_3> cleaned_vertices; // after extracting geometries, process the vertices and store cleaned vertices for one shell
    std::vector<std::vector<unsigned long>> cleaned_faces; // after extracting geometries, process the face indices and store cleaned faces for one shell

    void visit(Nef_polyhedron::Vertex_const_handle v) {}
    void visit(Nef_polyhedron::Halfedge_const_handle he) {}
    void visit(Nef_polyhedron::SHalfedge_const_handle she) {}
    void visit(Nef_polyhedron::SHalfloop_const_handle shl) {}
    void visit(Nef_polyhedron::SFace_const_handle sf) {}

    void visit(Nef_polyhedron::Halffacet_const_handle hf) {
        // do something to each half-facet of a shell
        for (Nef_polyhedron::Halffacet_cycle_const_iterator it = hf->facet_cycles_begin(); it != hf->facet_cycles_end(); it++) {

            //std::cout << it.is_shalfedge() << " " << it.is_shalfloop() << '\n';
            Nef_polyhedron::SHalfedge_const_handle she = Nef_polyhedron::SHalfedge_const_handle(it);
            CGAL_assertion(she != 0);
            Nef_polyhedron::SHalfedge_around_facet_const_circulator hc_start = she;
            Nef_polyhedron::SHalfedge_around_facet_const_circulator hc_end = hc_start;
            //std::cout << "hc_start = hc_end? " << (hc_start == hc_end) << '\n';

            faces.emplace_back();
            unsigned long index = 0; // index for each half-facet
            CGAL_For_all(hc_start, hc_end) // each vertex of one halffacet
            {
                Nef_polyhedron::SVertex_const_handle svert = hc_start->source();
                Point_3 vpoint = svert->center_vertex()->point();
                //std::cout << "v: " << "(" << vpoint.x() << ", " << vpoint.y() << ", " << vpoint.z() << ")" << '\n';
                vertices.push_back(vpoint);
                faces.back().push_back(index++);
            }
            //std::cout << '\n';

        }

    }
};



/*
* class to process Nef
* (1)extract geometries of a nef polyhedron
* (2)process extracted geometries for writing to cityjson
* (3)3D Minkowski sum
*/
class NefProcessing
{
public:
    /*
    * Extract geometries from a Nef polyhedron
    * @params:
    * nef: a Nef_Polyhedron
    * shell_explorers: store all the shells of one Nef_Polyhedron
    */
    static void extract_nef_geometries(const Nef_polyhedron& nef, std::vector<Shell_explorer>& shell_explorers)
    {
        std::cout << "extracting nef geometries ...\n";
        
        int volume_count = 0; // for counting volumes
        Nef_polyhedron::Volume_const_iterator current_volume;
        CGAL_forall_volumes(current_volume, nef)
        {
            //std::cout << "volume: " << volume_count++ << " ";
            //std::cout << "volume mark: " << current_volume->mark() << '\n';
            Nef_polyhedron::Shell_entry_const_iterator current_shell;
            CGAL_forall_shells_of(current_shell, current_volume)
            {
                Shell_explorer se;
                Nef_polyhedron::SFace_const_handle sface_in_shell(current_shell);
                nef.visit_shell_objects(sface_in_shell, se);

                // add the se to shell_explorers
                shell_explorers.emplace_back(se);
            }
        }

        // prompt some info
        /*std::cout << "after extracting geometries: " << '\n';
        std::cout << "shell explorers size: " << shell_explorers.size() << '\n';
        std::cout << "info for each shell\n";
        for (const auto& se : shell_explorers)
        {
            std::cout << "vertices size of this shell: " << se.vertices.size() << '\n';
            std::cout << "faces size of this shell: " << se.faces.size() << '\n';
            std::cout << '\n';
        }*/

        std::cout << "done\n";
    }


    /*
    * in Shell_explorer, the index is calculated for each half-facet
    * thus we need further processing the indices for writing to json file
    * i.e.
    * for one shell, its faces vector can be: [[0,1,2,3], [0,1,2,3], [0,1,2,3], [0,1,2,3]] after extracting geometries
    * since in visit(Nef_polyhedron::Halffacet_const_handle hf) function, we store the indices for each half-facet
    * thus when we start to visit a new half-facet, the index starts from 0, that means the indices are not accumulated for the whole shell
    * so after the process, the faces vector should be: [[0,1,2,3], [4,5,6,7],...] - this kind of indices are "desired" for writing cityjson file
    * and what's more, the repeatness needs to be taken into consideration
    *
    * @param:
    * shell_explorers: contains all shells for one building / solid
    * @output of this function:
    * if this function is called ->
    * Shell_explorer.cleaned_vertices will be filled with cleaned vertices (ideally no repeated vertices)
    * Shell_explorer.cleaned_faces will be filled with corresponding indices in cleaned_vertices vector (0-based)
    * cleaned_vertices and cleaned_faces are used for writing to json file without no repeated vertices warnings
    */
    static void process_shells_for_cityjson(std::vector<Shell_explorer>& shell_explorers)
    {
        std::cout << "processing shells for cityjson ..." << '\n';

        // step 1
        // get all vertices of all shells and corresponding face indices of each shell ----------------------
        // first store all the vertices in a vector
        std::vector<Point_3> all_vertices; // contains repeated vertices - for all shells in shell_explorers
        for (auto const& se : shell_explorers) {
            for (auto const& v : se.vertices) {
                all_vertices.push_back(v);
            }
        }

        // next store the face indices(accumulated from 0)
        unsigned long index_in_all_vertices = 0;
        for (auto& se : shell_explorers) {
            for (auto& face : se.faces) {
                for (auto& index : face) {
                    index = index_in_all_vertices++;
                }
            }
        }
        // now we have the all_vertices and face indices to write to cityjson -------------------------------


        // step 2
        // assume there are no repeated faces (repeated faces can exist principally)
        // cope with repeatness, get the cleaned_vertices and cleaned_faces ---------------------------------
        for (auto& se : shell_explorers) {
            unsigned long cleaned_index = 0; // indices of cleaned_faces, accumulated from 0, starts from each shell
            for (auto const& face : se.faces) {
                se.cleaned_faces.emplace_back(); // add a new empty face
                for (auto const& index : face) {
                    Point_3& vertex = all_vertices[index]; // get the vertex according to the index
                    if (!vertex_exist_check(se.cleaned_vertices, vertex)) { // if the vertex is not in cleaned_vertices vector
                        se.cleaned_vertices.push_back(vertex);
                        se.cleaned_faces.back().push_back(cleaned_index++); // add the cleaned_index to the newest added face, and increment the index
                    }
                    else { // if the vertex is already in the cleaned_vertices vector
                        unsigned long exist_vertex_index = find_vertex_index(se.cleaned_vertices, vertex);
                        se.cleaned_faces.back().push_back(exist_vertex_index); // add the found index to the newest added face
                    }
                }
            }
        }
        // now we have cleaned_vertices and cleaned_faces to write to cityjson ------------------------------
        
        std::cout << "done" << '\n';

    }



    /*
    * make a cube (type: Polyhedron) with side length: size
    * @param: 
    * size -> indicating the side length of the cube, default value is set to 0.1
    * @return:
    * Nef_polyhedron
    */
    static Nef_polyhedron make_cube(double size = 0.1)
    {
        Polyhedron_builder<Polyhedron::HalfedgeDS> polyhedron_builder; // used for create a cube

        // construct a cube with side length: size

        // (1) the front surface, vertices in CCW order(observing from outside):
        polyhedron_builder.faces.emplace_back();

        polyhedron_builder.vertices.emplace_back(Point_3(size, 0, 0)); // vertex index: 0
        polyhedron_builder.vertices.emplace_back(Point_3(size, size, 0)); // vertex index: 1
        polyhedron_builder.vertices.emplace_back(Point_3(size, size, size)); // vertex index: 2
        polyhedron_builder.vertices.emplace_back(Point_3(size, 0, size)); // vertex index: 3

        polyhedron_builder.faces.back() = { 0, 1, 2, 3 };

        // (2) the back surface, vertices in CCW order(observing from outside):
        polyhedron_builder.faces.emplace_back();

        polyhedron_builder.vertices.emplace_back(Point_3(0, size, 0)); // vertex index: 4
        polyhedron_builder.vertices.emplace_back(Point_3(0, 0, 0)); // vertex index: 5
        polyhedron_builder.vertices.emplace_back(Point_3(0, 0, size)); // vertex index: 6
        polyhedron_builder.vertices.emplace_back(Point_3(0, size, size)); // vertex index: 7

        polyhedron_builder.faces.back() = { 4, 5, 6, 7 };

        // after front and back surface, we now have all 8 vertices of a cube
        // repeatness should be avoided when adding vertices and faces to a polyhedron_builder

        // (3) the top surface, vertices in CCW order(observing from outside):
        polyhedron_builder.faces.emplace_back();

        //Point_3(1, 0, 1); // vertex index: 3
        //Point_3(1, 1, 1); // vertex index: 2
        //Point_3(0, 1, 1); // vertex index: 7
        //Point_3(0, 0, 1); // vertex index: 6

        polyhedron_builder.faces.back() = { 3, 2, 7, 6 };

        // (4) the down surface, vertices in CCW order(observing from outside):
        polyhedron_builder.faces.emplace_back();

        //Point_3(0, 0, 0); // vertex index: 5
        //Point_3(0, 1, 0); // vertex index: 4
        //Point_3(1, 1, 0); // vertex index: 1
        //Point_3(1, 0, 0); // vertex index: 0

        polyhedron_builder.faces.back() = { 5, 4, 1, 0 };

        // (5) the left surface, vertices in CCW order(observing from outside):
        polyhedron_builder.faces.emplace_back();

        //Point_3(0, 0, 0); // vertex index: 5
        //Point_3(1, 0, 0); // vertex index: 0
        //Point_3(1, 0, 1); // vertex index: 3
        //Point_3(0, 0, 1)); // vertex index: 6

        polyhedron_builder.faces.back() = { 5, 0, 3, 6 };

        // (6) the right surface, vertices in CCW order(observing from outside):
        polyhedron_builder.faces.emplace_back();

        //Point_3(1, 1, 0); // vertex index: 1
        //Point_3(0, 1, 0); // vertex index: 4
        //Point_3(0, 1, 1); // vertex index: 7
        //Point_3(1, 1, 1); // vertex index: 2

        polyhedron_builder.faces.back() = { 1, 4, 7, 2 };

        // now build the Polyhedron
        Polyhedron cube;
        cube.delegate(polyhedron_builder);

        // check whether the cube is correctly created
        if (cube.is_empty()) { // if the cube is empty  
            std::cerr << "warning: created an empty cube, please check make_cube() function in Polyhedron.hpp\n";
            return Nef_polyhedron::EMPTY; // return an empty nef polyhedron
        }
           
        if (!cube.is_closed()) { // if the cube is NOT closed
            std::cerr << "warning: cube(Polyhedron) is not closed, please check make_cube() function in Polyhedron.hpp\n";
            return Nef_polyhedron::EMPTY; // return an empty nef polyhedron
        }
                  
        // cube is correctly created(not empty && closed) -> convert it to a Nef_polyhedron
        Nef_polyhedron nef_cube(cube);
        
        return nef_cube;

    }



    /*
    * 3D Minkowski sum
    * details: https://doc.cgal.org/latest/Minkowski_sum_3/index.html#Chapter_3D_Minkowski_Sum_of_Polyhedra
    * 
    * @param
    * nef : the nef polyhedron which needs to be merged
    * size: a cube's side length
    */
    static Nef_polyhedron minkowski_sum(Nef_polyhedron& nef, double size = 0.1)
    {
        Nef_polyhedron cube = make_cube(size);
        return CGAL::minkowski_sum_3(nef, cube);     
    }



protected:
    /*
    * The vertex_exist_check() and find_vertex_index() fucntions are also in JsonHandler class
    * there may be other ways to avoid code repeatness
    * for now, let's just define and use them
    */


    /*
    * check if a vertex already exists in a vertices vector
    * use coordinates to compare whether two vertices are the same
    * return: False - not exist, True - already exist
    */
    static bool vertex_exist_check(std::vector<Point_3>& vertices, const Point_3& vertex) {
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
    static unsigned long find_vertex_index(std::vector<Point_3>& vertices, Point_3& vertex) {
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


};



/*
* ------------------------------------------------------------------------------------------------------------------------------------------
* now we have finished building big nef polyhedron
* there are some possible post processing steps
* (1) regularization
* (2) erosion
* ------------------------------------------------------------------------------------------------------------------------------------------
*/



/*
* remeshing helper
*/
struct halfedge2edge
{
    halfedge2edge(const Mesh& m, std::vector<edge_descriptor>& edges)
        : m_mesh(m), m_edges(edges)
    {}
    void operator()(const halfedge_descriptor& h) const
    {
        m_edges.push_back(edge(h, m_mesh));
    }
    const Mesh& m_mesh;
    std::vector<edge_descriptor>& m_edges;
};



/*
* class to perform possible post processing steps for Nef polyhedron
* possible steps for now:
* (1) regularization
* (2) erosion 
*/
class PostProcesssing {
public:

    /*
    * regularization
    * returns the regularized polyhedron (closure of the interior)
    */
    Nef_polyhedron get_regularized_nef(Nef_polyhedron& nef) {
        Nef_polyhedron regularized_nef = nef.regularization();
        return regularized_nef;
    }



    /*
    * get bounding box for Nef polyhedron
    * use a user-defined size (10 units by default)
    * 
    * this function is inspired or just small modifications based on val3dity:
    * see: https://github.com/tudelft3d/val3dity/blob/main/src/geomtools.cpp#L210
    */
    static Nef_polyhedron get_nef_bbox(Nef_polyhedron& nef, double size = 10) {
        
        double xmin = 1e12;
        double ymin = 1e12;
        double zmin = 1e12;
        double xmax = -1e12;
        double ymax = -1e12;
        double zmax = -1e12;

        Nef_polyhedron::Vertex_const_iterator v;
        for (v = nef.vertices_begin(); v != nef.vertices_end(); v++)
        {
            if (CGAL::to_double(v->point().x()) - xmin < epsilon)
                xmin = CGAL::to_double(v->point().x());

            if (CGAL::to_double(v->point().y()) - ymin < epsilon)
                ymin = CGAL::to_double(v->point().y());

            if (CGAL::to_double(v->point().z()) - zmin < epsilon)
                zmin = CGAL::to_double(v->point().z());

            if (CGAL::to_double(v->point().x()) - xmax > epsilon)
                xmax = CGAL::to_double(v->point().x());

            if (CGAL::to_double(v->point().y()) - ymax > epsilon)
                ymax = CGAL::to_double(v->point().y());

            if (CGAL::to_double(v->point().z()) - zmax > epsilon)
                zmax = CGAL::to_double(v->point().z());
        }

        //-- expand the bbox by size units
        xmin -= size;
        ymin -= size;
        zmin -= size;
        xmax += size;
        ymax += size;
        zmax += size;

        //-- write an OFF file and convert Nef, simplest (and fastest?) solution
        std::stringstream ss;
        ss << "OFF" << std::endl
            << "8 6 0" << std::endl
            << xmin << " " << ymin << " " << zmin << std::endl
            << xmax << " " << ymin << " " << zmin << std::endl
            << xmax << " " << ymax << " " << zmin << std::endl
            << xmin << " " << ymax << " " << zmin << std::endl
            << xmin << " " << ymin << " " << zmax << std::endl
            << xmax << " " << ymin << " " << zmax << std::endl
            << xmax << " " << ymax << " " << zmax << std::endl
            << xmin << " " << ymax << " " << zmax << std::endl
            << "4 0 3 2 1" << std::endl
            << "4 0 1 5 4" << std::endl
            << "4 1 2 6 5" << std::endl
            << "4 2 3 7 6" << std::endl
            << "4 0 4 7 3" << std::endl
            << "4 4 5 6 7" << std::endl;

        Nef_polyhedron nefbbox;
        CGAL::OFF_to_nef_3(ss, nefbbox);
        return nefbbox;
    }



    /*
    * erosion of a Nef
    * (1) get bounding box
    * (2) get its complement
    * (3) enlarge the complement -> using minkowski sum
    * (4) eroded_nef = nef - enlarged_complement? or bbox - enlarged_complement?
    */
    static Nef_polyhedron get_eroded_nef(Nef_polyhedron& nef, double minkowski_param = 0.01) {

        Nef_polyhedron nefbbox = get_nef_bbox(nef);
        Nef_polyhedron complement = nefbbox - nef;
        Nef_polyhedron tmp = NefProcessing::minkowski_sum(complement, minkowski_param);
        Nef_polyhedron eroded_nef = nef - tmp;

        // regularization?

        return eroded_nef;
    }



    /*
    * remeshing
    */
    static void remeshing(Nef_polyhedron& big_nef, const std::string& file, double target_edge_length) {
        Polyhedron polyhedron;

        if (!big_nef.is_simple()) {
            std::cerr << "big nef is not simple, can not convert to polyhedron, please check" << '\n';
            return;
        }

        // convert big_nef to polyhedron
        big_nef.convert_to_polyhedron(polyhedron);

        // convert to surface mesh
        Mesh mesh;
        CGAL::copy_face_graph(polyhedron, mesh);
        std::cout << "is polygon mesh valid? " << mesh.is_valid() << '\n';

        // remeshing
        unsigned int nb_iter = 3;
        std::cout << "target edge length: " << target_edge_length << '\n';
        std::cout << "Split border...";
        std::vector<edge_descriptor> border;
        PMP::border_halfedges(faces(mesh), mesh, boost::make_function_output_iterator(halfedge2edge(mesh, border)));
        PMP::split_long_edges(border, target_edge_length, mesh);
        std::cout << "done." << '\n';
        std::cout << "Start remeshing of " << " (" << num_faces(mesh) << " faces)..." << '\n';
        PMP::isotropic_remeshing(faces(mesh), target_edge_length, mesh,
            CGAL::parameters::number_of_iterations(nb_iter)
            .protect_constraints(true)); //i.e. protect border, here
        std::cout << "Remeshing done." << '\n';

        // output
        std::ofstream out_stream(file);
        out_stream.precision(17); // why use 17? from CGAL docs setting precisions can reduce the self-intersection errors in the output
        //bool status = CGAL::IO::write_OFF(out_stream, mesh);
        out_stream.close();
        //if (status)std::cout << "file saved at: " << file << '\n';
        
    }


};