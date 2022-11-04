import numpy as np
import pyvista as pv
from shapely.geometry import Polygon, LineString
from helpers.geometry import *
from helpers.cluster import *


def symmetric_difference(meshes):
    """Return the intersection between the surfaces of multiple meshes"""
    
    n_meshes = len(meshes)
    
    dif_polys = []


    # labels = ndarray of cluster labels associated with each face, n_clusters = integer [number of clusters]
    labels, n_clusters = cluster_meshes(meshes)

    all_idxs = [[],[]]
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
            all_idxs[0].append(idxs[0])
            all_idxs[1].append(idxs[1])
            project_diff_to_3d(polys[0], polys[1], origin, normal, dif_polys)
    return all_idxs, dif_polys
                                               


def project_diff_to_3d(polys_1: Polygon, polys_2: Polygon, origin: np.ndarray, normal: np.ndarray, dif_polys) -> None:
        """Calculate symmetric difference, project into 3d and store in list dif_polys"""
        dif1 = polys_1.difference(polys_2)
        dif2 = polys_2.difference(polys_1)
        
        #Differences are calculated
        if dif1.area > 0.001:
            if dif1.type == "MultiPolygon":
                for geom in dif1.geoms:
                    if geom.area > 0.01:
                        if len(list(geom.interiors)) > 0: #Polygon has a hole. We cut the polygon along the hole to remove it.
                            for interior in geom.interiors:
                                cut = LineString([(geom.bounds[0], interior.coords[0][1]), (geom.bounds[2], interior.coords[0][1])])
                                cut = cut.buffer(0.001)
                                geomcut = geom.difference(cut)
                                for piece in geomcut.geoms:
                                    pts = to_3d(piece, normal, origin)
                                    common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                          
                                    common_mesh_dif = common_mesh_dif.triangulate()
                                    common_mesh_dif.flip_normals() 
                                    dif_polys.append(common_mesh_dif) 
                        else:
                            pts = to_3d(geom, normal, origin)  
                            common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                        
                            common_mesh_dif = common_mesh_dif.triangulate()
                            common_mesh_dif.flip_normals() 
                            dif_polys.append(common_mesh_dif)
                    
            if dif1.type == "Polygon":
                if len(list(dif1.interiors)) > 0: #Polygon has a hole. We cut the polygon along the hole to remove it.
                        for interior in dif1.interiors:
                            cut = LineString([(dif1.bounds[0], interior.coords[0][1]), (dif1.bounds[2], interior.coords[0][1])])
                            cut = cut.buffer(0.001)
                            geomcut = dif1.difference(cut)
                            for piece in geomcut.geoms:
                                pts = to_3d(piece, normal, origin)
                                common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                          
                                common_mesh_dif = common_mesh_dif.triangulate()
                                common_mesh_dif.flip_normals() 
                                dif_polys.append(common_mesh_dif) 
                else:
                    pts = to_3d(dif1, normal, origin)  
                    common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                          
                    common_mesh_dif = common_mesh_dif.triangulate()
                    common_mesh_dif.flip_normals() 
                    dif_polys.append(common_mesh_dif)
                    
        if dif2.area > 0.001:
            if dif2.type == "MultiPolygon":
                for geom in dif2.geoms:
                    if geom.area > 0.01:
                        if len(list(geom.interiors)) > 0: #Polygon has a hole. We cut the polygon along the hole to remove it.
                            for interior in geom.interiors:
                                cut = LineString([(geom.bounds[0], interior.coords[0][1]), (geom.bounds[2], interior.coords[0][1])])
                                cut = cut.buffer(0.001)
                                geomcut = geom.difference(cut)
                                for piece in geomcut.geoms:
                                    pts = to_3d(piece, normal, origin)
                                    common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                          
                                    dif_polys.append(common_mesh_dif) 
                        else:
                            pts = to_3d(geom, normal, origin)  
                            common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                        
                            dif_polys.append(common_mesh_dif)
                    
            if dif2.type == "Polygon":
                if len(list(dif2.interiors)) > 0: #Polygon has a hole. We cut the polygon along the hole to remove it.
                        for interior in dif2.interiors:
                            cut = LineString([(dif2.bounds[0], interior.coords[0][1]), (dif2.bounds[2], interior.coords[0][1])])
                            cut = cut.buffer(0.001)
                            geomcut = dif2.difference(cut)
                            for piece in geomcut.geoms:
                                pts = to_3d(piece, normal, origin)
                                common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                          
                                dif_polys.append(common_mesh_dif) 
                else:
                    pts = to_3d(dif2, normal, origin)  
                    common_mesh_dif = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))), n_faces=len(pts))                          
                    dif_polys.append(common_mesh_dif)
        return None