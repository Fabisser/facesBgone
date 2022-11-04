#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>
#include <CGAL/Polygon_mesh_processing/border.h>
#include <CGAL/Polygon_mesh_processing/IO/polygon_mesh_io.h>
#include <CGAL/Polygon_mesh_processing/polygon_mesh_to_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/repair_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>
#include <boost/lexical_cast.hpp>
#include <iostream>
#include <iterator>
#include <string>
#include <tuple>
#include <vector>
typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::FT                                                   FT;
typedef std::vector<std::size_t>                                CGAL_Polygon;
typedef std::array<FT, 3>                                       Custom_point;
typedef Kernel::Point_3                                     Point;
typedef CGAL::Surface_mesh<Point>                           Mesh;
typedef boost::graph_traits<Mesh>::vertex_descriptor        vertex_descriptor;
typedef boost::graph_traits<Mesh>::halfedge_descriptor      halfedge_descriptor;
typedef boost::graph_traits<Mesh>::face_descriptor          face_descriptor;
namespace PMP = CGAL::Polygon_mesh_processing;

// For polygon soup to be created
struct Array_traits
{
  struct Equal_3
  {
    bool operator()(const Custom_point& p, const Custom_point& q) const {
      return (p == q);
    }
  };
  struct Less_xyz_3
  {
    bool operator()(const Custom_point& p, const Custom_point& q) const {
      return std::lexicographical_compare(p.begin(), p.end(), q.begin(), q.end());
    }
  };
  Equal_3 equal_3_object() const { return Equal_3(); }
  Less_xyz_3 less_xyz_3_object() const { return Less_xyz_3(); }
};

// Incrementally fill the holes that are no larger than given diameter
// and with no more than a given number of edges (if specified).
bool is_small_hole(halfedge_descriptor h, Mesh & mesh,
                   double max_hole_diam, int max_num_hole_edges)
{
  int num_hole_edges = 0;
  CGAL::Bbox_3 hole_bbox;
  for (halfedge_descriptor hc : CGAL::halfedges_around_face(h, mesh))
  {
    const Point& p = mesh.point(target(hc, mesh));
    hole_bbox += p.bbox();
    ++num_hole_edges;
    // Exit early, to avoid unnecessary traversal of large holes
    if (num_hole_edges > max_num_hole_edges) return false;
    if (hole_bbox.xmax() - hole_bbox.xmin() > max_hole_diam) return false;
    if (hole_bbox.ymax() - hole_bbox.ymin() > max_hole_diam) return false;
    if (hole_bbox.zmax() - hole_bbox.zmin() > max_hole_diam) return false;
  }
  return true;
}


int main(int argc, char* argv[])
{


  std::string filepath = DATA_PATH + std::string("/dataset_1.stl");
  const std::string filename = (argc > 1) ? argv[1] : CGAL::data_file_path(filepath);
  Mesh mesh;
  if(!PMP::IO::read_polygon_mesh(filename, mesh))
  {
    std::cerr << "Invalid input." << std::endl;
    return 1;
  }
  // From Polygon_mesh_processing/hole_filling_example_SM.cpp


  // Both of these must be positive in order to be considered
  double max_hole_diam   = (argc > 2) ? boost::lexical_cast<double>(argv[2]): -1;
  int max_num_hole_edges = (argc > 3) ? boost::lexical_cast<int>(argv[3]) : -1;
  unsigned int nb_holes = 0;
  std::vector<halfedge_descriptor> border_cycles;
  // collect one halfedge per boundary cycle
  PMP::extract_boundary_cycles(mesh, std::back_inserter(border_cycles));
  
  for(halfedge_descriptor h : border_cycles)
  {
    if(max_hole_diam > 0 && max_num_hole_edges > 0 &&
       !is_small_hole(h, mesh, max_hole_diam, max_num_hole_edges))
      continue;
    std::vector<face_descriptor>  patch_facets;
    std::vector<vertex_descriptor> patch_vertices;
    PMP::triangulate_and_refine_hole(mesh,
                                        h,
                                        std::back_inserter(patch_facets),
                                        std::back_inserter(patch_vertices));
    ++nb_holes;
  }
  std::cout << std::endl;
  std::cout << nb_holes << " holes have been filled" << std::endl;

  // Try some mesh fixing with CGAL
  std::vector<std::array<FT, 3> > points;
  std::vector<CGAL_Polygon> polygons;
  Mesh finalmesh;

  PMP::polygon_mesh_to_polygon_soup(mesh, points, polygons);
  PMP::repair_polygon_soup(points, polygons, CGAL::parameters::geom_traits(Array_traits()));
  PMP::orient_polygon_soup(points, polygons);
  PMP::polygon_soup_to_polygon_mesh(points, polygons, finalmesh);

  // Output
  CGAL::IO::write_polygon_mesh("dataset_1_output.stl", finalmesh, CGAL::parameters::stream_precision(17));
  std::cout << "Final Mesh written to: ..." << std::endl;
  return 0;
}