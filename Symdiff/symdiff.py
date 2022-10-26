
import numpy as np
import pyvista as pv
from shapely.geometry import Polygon, LineString
from helpers.geometry import *
from helpers.cluster import *


def symmetric_difference(meshes: list[pv.PolyData]) -> tuple[list[list[int]], list[pv.PolyData]]:
    """Return the intersection between the surfaces of multiple meshes"""
    
    n_meshes = len(meshes)
    
    areas_symdif = []


    # labels = ndarray of cluster labels associated with each face, n_clusters = integer [number of clusters]
    labels, n_clusters = cluster_meshes(meshes)

    all_idxs = []
    # for every cluster
    for plane in range(n_clusters):
        # For every mesh, extract the index of the faces that belong to this cluster
        # idxs is a list of sublists (one for each mesh) containing face indices
        idxs = [[i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)]

        # check to ensure that both meshes have at least one face that belongs to this cluster
        if any([len(idx) == 0 for idx in idxs]):
            continue         
        
        # take surfaces from each mesh that belong to this cluster, and put them in a polydata object
        msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes)]
        
        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        # get the normal of the first face of the first mesh
        normal = msurfaces[0].face_normals[0]
        
        
        # Create the two 2D polygons by projecting the faces
        
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]
        # Intersect the 2D polygons
        inter = polys[0].intersection(polys[1])
        
        if inter.area > 0.1:
            # project to 2d
            all_idxs.append(idxs[0])
            project_symdiff_to_3d(polys[0], polys[1], origin, normal, areas_symdif)
    return all_idxs, areas_symdif
                                               


def project_symdiff_to_3d(polys_1: Polygon, polys_2: Polygon, origin: np.ndarray, normal: np.ndarray, areas_symdif: list[pv.PolyData]) -> None:
        """Calculate symmetric difference, project into 3d and store in list areas_symdif"""
        symdif = polys_1.symmetric_difference(polys_2)
        if symdif.area > 0.001:
            if symdif.type == "MultiPolygon":
                for geom in symdif.geoms:
                    if geom.area > 0.01:
                        if len(list(geom.interiors)) > 0: #Polygon has a hole. We cut the polygon along the hole to remove it.
                            for interior in geom.interiors:
                                cut = LineString([(geom.bounds[0], interior.coords[0][1]), (geom.bounds[2], interior.coords[0][1])])
                                cut = cut.buffer(0.001)
                                geomcut = geom.difference(cut)
                                for piece in geomcut.geoms:
                                    pts = to_3d(piece, normal, origin)
                                    common_mesh_symdif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))                          

                                    common_mesh_symdif["area"] = [symdif.area]
                                    areas_symdif.append(common_mesh_symdif) 
                        else:
                            pts = to_3d(geom, normal, origin)  
                            common_mesh_symdif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))                        
                            common_mesh_symdif["area"] = [symdif.area]
                            areas_symdif.append(common_mesh_symdif)
                    
            if symdif.type == "Polygon":
                if len(list(symdif.interiors)) > 0: #Polygon has a hole. We cut the polygon along the hole to remove it.
                        for interior in symdif.interiors:
                            cut = LineString([(symdif.bounds[0], interior.coords[0][1]), (symdif.bounds[2], interior.coords[0][1])])
                            cut = cut.buffer(0.001)
                            geomcut = symdif.difference(cut)
                            for piece in geomcut.geoms:
                                pts = to_3d(piece, normal, origin)
                                common_mesh_symdif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))                          
                                
                                common_mesh_symdif["area"] = [symdif.area]
                                areas_symdif.append(common_mesh_symdif) 
                else:
                    pts = to_3d(symdif, normal, origin)  
                    common_mesh_symdif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))                          
                    common_mesh_symdif["area"] = [symdif.area]
                    areas_symdif.append(common_mesh_symdif)
        return None