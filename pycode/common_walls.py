#this script only keeps what is essential to finding common walls

import json
import numpy as np
import pyvista as pv
import math
import os
from .helpers.geometry import surface_normal, project_2d, axes_of_normal
from shapely.geometry import MultiPolygon, Polygon
import scipy
from sklearn.cluster import AgglomerativeClustering
import rtree.index
from . import cityjson
import matplotlib.pyplot as plt


#compute surface normal (might be useful)
def surface_normal(poly):
    n = [0.0, 0.0, 0.0]

    for i, v_curr in enumerate(poly):
        v_next = poly[(i+1) % len(poly)]
        n[0] += (v_curr[1] - v_next[1]) * (v_curr[2] + v_next[2])
        n[1] += (v_curr[2] - v_next[2]) * (v_curr[0] + v_next[0])
        n[2] += (v_curr[0] - v_next[0]) * (v_curr[1] + v_next[1])
    
        if all([c == 0 for c in n]):
            raise ValueError("No normal. Possible colinear points!")

    normalised = [i/np.linalg.norm(n) for i in n]

    return normalised


def is_on_plane(point, normal, origin):
    a, b, c, d = get_plane_params(normal, origin)
    
    x, y, z = point
    
    return a * x + b * y + c * z + d == 0


def plane_params(normal, origin, rounding=2, absolute=True):
    """Returns the params (a, b, c, d) of the plane equation"""
    a, b, c = np.round_(normal, 3)
    x0, y0, z0 = origin
    
    d = -(a * x0 + b * y0 + c * z0)
    
    if rounding >= 0:
        d = round(d, rounding)
    
    return np.array([a, b, c, d])


def face_planes(mesh):
    return [plane_params(mesh.face_normals[i], mesh.cell_points(i)[0]) for i in range(mesh.n_cells)]


def project_mesh(mesh, normal, origin):
    p = []
    for i in range(mesh.n_cells):
        pts = mesh.cell_points(i)
        
        pts_2d = project_2d(pts, normal, origin)
        
        p.append(Polygon(pts_2d))
    
    return MultiPolygon(p).buffer(0)


def to_3d(polygon, normal, origin):
    xa, ya = axes_of_normal(normal)

    mat = np.array([xa, ya])
    pts = np.array(polygon.boundary.coords)

    return np.dot(pts, mat) + origin


def cluster_faces(data, threshold=0.1):
    ndata = np.array(data)
    
    dm1 = scipy.spatial.distance_matrix(ndata, ndata)
    dm2 = scipy.spatial.distance_matrix(ndata, -ndata)

    distance_matrix = np.minimum(dm1, dm2)

    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold, affinity='precomputed', linkage='average').fit(distance_matrix)
    
    return clustering.labels_, clustering.n_clusters_


def cluster_meshes(meshes, threshold=0.1):
    """Clusters the faces of the given meshes"""
    
    n_meshes = len(meshes)
    
    # Compute the "absolute" plane params for every face of the two meshes
    # list containing sublist for each mesh (each sublist containing one array for parameters of each plane)
    planes = [face_planes(mesh) for mesh in meshes]
    mesh_ids = [[m for _ in range(meshes[m].n_cells)] for m in range(n_meshes)]
    
    # Find the common planes between the two meshes
    # array of parameters for all planes in both meshes (combined)
    all_planes = np.concatenate(planes)
    all_labels, n_clusters = cluster_faces(all_planes, threshold)
    areas = []

    # list of arrays (one for each mesh) indicating cluster labels for each face
    labels = np.array_split(all_labels, [meshes[m].n_cells for m in range(n_meshes - 1)])
    
    return labels, n_clusters


def intersect_surfaces(meshes):
    """Return the intersection between the surfaces of multiple meshes"""
    
    n_meshes = len(meshes)
    
    areas = []

    # labels = ndarray of cluster labels associated with each face, n_clusters = integer [number of clusters]
    labels, n_clusters = cluster_meshes(meshes)

    # for every cluster
    for plane in range(n_clusters):
        # For every mesh, extract the index of the faces that belong to this cluster
        # idxs is a list of sublists (one for each mesh) containing face indices
        idxs = [[i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)]

        # check to ensure that both meshes have at least one face that belongs to this cluster
        if any([len(idx) == 0 for idx in idxs]):
            continue
        # Logic says: if any of the faces of the different meshes, are found to belong to the same cluster
        # it means that the buildings are adjacent
        # and I don't need the rest of the code of this function --> which leads to errors
        else:
            return True
        
    return False                
        
# """ OLD CODE for intersect_surfaces function """

#         # take surfaces from each mesh that belong to this cluster, and put them in a polydata object
#         msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes)]
                
#         # Set the normal and origin point for a plane to project the faces
#         origin = msurfaces[0].clean().points[0]
#         # get the normal of the first face of the first mesh
#         normal = msurfaces[0].face_normals[0]
        
#         # Create the two 2D polygons by projecting the faces
#         # creates list of polygons and multipolygons (one for each mesh)
#         polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]
        
#         # Intersect the 2D polygons => finds intersection of all polygons
#         inter = polys[0]
#         for i in range(1, len(polys)):
#             inter = inter.intersection(polys[i])
        
#         if inter.area > 0.001:
#             if inter.type == "MultiPolygon":
#                 #  project back to 3D
#                 for geom in inter.geoms:
#                     pts = to_3d(geom, normal, origin)
#                     common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
#                     common_mesh["area"] = [geom.area]
#                     areas.append(common_mesh)
#             else:
#                     pts = to_3d(inter, normal, origin)
#                     common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
#                     common_mesh["area"] = [inter.area]
#                     areas.append(common_mesh)
    
#     return areas


def intersect_pairs(mesh, neighbours):
    # can call this function to get common walls between one mesh and many others
    return np.hstack([intersect_surfaces([mesh, neighbour]) for neighbour in neighbours])


def get_bbox(geom, verts):
    pts = np.array(cityjson.get_points(geom, verts))

    return np.hstack([[np.min(pts[:, i]), np.max(pts[:, i])] for i in range(np.shape(pts)[1])])


def generator_function(cm, verts):
    for i, objid in enumerate(cm["CityObjects"]):
        obj = cm["CityObjects"][objid]
        xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(obj["geometry"][0], verts)
        yield (i, (xmin, ymin, zmin, xmax, ymax, zmax), objid)


def rpath(path):
    return os.path.expanduser(path)


def plot_meshes(meshes, **kargs):
    p = pv.Plotter(**kargs)

    p.add_mesh(meshes[0], color="red")
    for mesh in meshes[1:]:
        p.add_mesh(mesh)

    p.show()


def distance(x, y):
    """Returns the euclidean distance between two points"""

    return math.sqrt(sum([math.pow(x[c] - y[c], 2) for c in range(len(x))]))


def abs_distance(x, y):
    """Returns the minimum absolute distance"""

    return min(distance(x, y), distance(x, [-e for e in y]))



# Implementation

def common_walls(path):
    # Load cityjson
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    filename = path

    with open(filename) as file:
        cm = json.load(file)

    if "transform" in cm:
        s = cm["transform"]["scale"]
        t = cm["transform"]["translate"]
        verts = [[v[0] * s[0] + t[0], v[1] * s[1] + t[1], v[2] * s[2] + t[2]]
                for v in cm["vertices"]]
    else:
        verts = cm["vertices"]

    # mesh points
    vertices = np.array(verts)

    # what is this?
    p = rtree.index.Property()
    p.dimension = 3
    r = rtree.index.Index(generator_function(cm, vertices), properties=p)


    # MULTIPLE BUILDING VERSION
    clustered_buildings = {}


    for building_part in cm['CityObjects']:

        # filter for only geometry objects
        if '-' in building_part:     

            clustered_buildings[building_part] = [] # add a key for every object navme and an empty list as its value 

            # get bounding box of reference building, find the objects that intersect its bbox
            xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(cm["CityObjects"][building_part]["geometry"][0], verts)
            objids = [n.object for n in r.intersection((xmin, ymin, zmin, xmax, ymax, zmax), objects=True) if n.object != building_part]

            main_mesh = cityjson.to_triangulated_polydata(cm["CityObjects"][building_part]["geometry"][2], vertices).clean()
            # meshes = [cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][2], vertices).clean() for objid in objids if '-' in objid]
            meshes = []
            meshes_id = []
            for objid in objids:
                if '-' in objid: # and objid in check_list:
                    # save the id of the object as well 
                    meshes_id.append(objid)
                    meshes.append(cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][2], vertices).clean())

            # Make origin of the building mesh at the center (mean) of the points
            t = np.mean(main_mesh.points, axis=0)
            main_mesh.points -= t
            for mesh in meshes:
                mesh.points -= t

            # visualize candidate buildings + reference building
            # plot_meshes([main_mesh] + meshes)
            # plot = main_mesh.plot(scalars=cluster_meshes([main_mesh])[0])

            # plot the intersections between surfaces
            intersections = np.array

            for nearby_building in range(len(meshes)):
                #plot_meshes([main_mesh] + [meshes[nearby_building]])
                intersection = intersect_surfaces([main_mesh, meshes[nearby_building]])
    #             if len(intersection) > 0:                  
                if intersection == True:   
                    # update the dictionary: add to the list saved for each object
                    clustered_buildings[building_part].append(meshes_id[nearby_building]) 


    #                 # show nearby building that has at lease one shared face
    #                 meshes[nearby_building].plot(scalars=cluster_meshes([meshes[nearby_building]])[0], text=f"Nearby building[{nearby_building}]")
    #                 # show each of the shared faces individually
    #                 for i in range(len(intersection)):
    #                     intersection[i].plot(text=f"Intersection[{i}]")
        

    print(clustered_buildings)    
            
    # with open('merge_buildings.txt', 'w') as f:
    #     for i in all_merged_buildings:
    #         f.write(";")
    #         for j in i:
    #             f.write(j+',')

    # main_mesh.save("cluster.vtk")


                
    for key in clustered_buildings: 
        # if clustered_buildings[key] != []: # the list 
            # if (key in clustered_buildings[key2] for key2 in clustered_buildings if key2!=key):
        for key2 in clustered_buildings:
            if key2!=key:
                # if the key I am iterating on is value in another key of the dictionary 
                if key in clustered_buildings[key2]: # the elements of the list            
                    clustered_buildings[key2] = clustered_buildings[key] + clustered_buildings[key2]                     
                    # then empty list of key2
                    clustered_buildings[key] = []
            

    list_adjb = []
    for key in clustered_buildings: 
        if clustered_buildings[key] != []: # the list 
            list_adjb.append(list(np.unique(clustered_buildings[key])))    
            
    return list_adjb