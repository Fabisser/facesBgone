#this script only keeps what is essential to finding common walls

import json
import numpy as np
import pyvista as pv
import math
import os
from helpers.geometry import surface_normal, project_2d, axes_of_normal
from shapely.geometry import MultiPolygon, Polygon
from shapely.validation import make_valid
import scipy
from sklearn.cluster import AgglomerativeClustering
import rtree.index
import cityjson
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
    planes = [face_planes(mesh) for mesh in meshes]
    mesh_ids = [[m for _ in range(meshes[m].n_cells)] for m in range(n_meshes)]
    
    # Find the common planes between the two meshes
    all_planes = np.concatenate(planes)
    all_labels, n_clusters = cluster_faces(all_planes, threshold)
    areas = []
    
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
        idxs = [[i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)]

        # check to ensure that both meshes have at least one face that belongs to this cluster
        if any([len(idx) == 0 for idx in idxs]):
            continue


        msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes)]
                
        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        normal = msurfaces[0].face_normals[0]
        
        # Create the two 2D polygons by projecting the faces
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]
        
        # Intersect the 2D polygons => finds intersection of all polygons
        union = polys[0]
        inter = polys[0].intersection(polys[1])
        for i in range(1, len(polys)):
            inter = inter.union(union.intersection(polys[i]))
            union = union.union(polys[i]) 
        print("UNION")
        print(union)
        print("INTERSETCION")
        print(inter)
        inter = union.difference(inter)
        print("difference")
        print(inter)
        if inter.area > 0.001:
            if inter.type == "MultiPolygon":
                #  project back to 3D
                for geom in inter.geoms:
                    print(geom.is_valid)
                    pts = to_3d(geom, normal, origin)
                    common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
                    common_mesh["area"] = [geom.area]
                    areas.append(common_mesh)
            elif inter.type == "Polygon":
                pts = to_3d(inter, normal, origin)
                common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
                common_mesh["area"] = [inter.area]
                areas.append(common_mesh)
    
    return areas


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

# Load cityjson
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

filename = "3dbag_v210908_fd2cee53_5907.json"

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

# SINGLE BUILDING VERSION
# # identify a reference building that will be compared to all the others
# mainid = "NL.IMBAG.Pand.0503100000030029-0"
# # get bounding box of reference building, find the objects that intersect its bbox
# xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(cm["CityObjects"][mainid]["geometry"][0], verts)
# objids = [n.object for n in r.intersection((xmin, ymin, zmin, xmax, ymax, zmax), objects=True) if n.object != mainid]
#
# main_mesh = cityjson.to_triangulated_polydata(cm["CityObjects"][mainid]["geometry"][0], vertices).clean()
# meshes = [cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][0], vertices).clean() for objid in objids]
#
# #what does this do?
# t = np.mean(main_mesh.points, axis=0)
# main_mesh.points -= t
# for mesh in meshes:
#     mesh.points -= t
#
# # visualizing results
#
# # plot the intersection between surfaces
# intersect_surfaces([main_mesh, meshes[0]])[0].plot()
#
# # viewing clusters, option 1
# plot_meshes([main_mesh] + meshes)
# plot = main_mesh.plot(scalars=cluster_meshes([main_mesh])[0])
# meshes[0].plot(scalars=cluster_meshes([meshes[0]])[0])
#
# # viewing clusters, option 2
# labels1, n_clusters = cluster_faces(face_planes(main_mesh))
# labels2, n_clusters2 = cluster_faces(face_planes(meshes[0]))
# boring_cmap = plt.cm.get_cmap("tab20b", n_clusters)
# main_mesh.plot(scalars=labels1, show_edges=True, cmap=boring_cmap)
# meshes[0].plot(scalars=labels2, show_edges=True, cmap=boring_cmap)
#
# main_mesh.save("cluster.vtk")

# MULTIPLE BUILDING VERSION
for building_part in cm['CityObjects']:
    # initiate list of buildings to merge eventually
    BAG_ids =[]
    merge_buildings = []
    # filter for only geometry objects
    if '-' in building_part:
        # get bounding box of reference building, find the objects that intersect its bbox
        xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(cm["CityObjects"][building_part]["geometry"][0], verts)
        objids = [n.object for n in r.intersection((xmin, ymin, zmin, xmax, ymax, zmax), objects=True) if n.object != building_part]

        main_mesh = cityjson.to_triangulated_polydata(cm["CityObjects"][building_part]["geometry"][2], vertices).clean()
        # meshes = [cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][2], vertices).clean() for objid in objids if '-' in objid]
        meshes = []
        for objid in objids:
            if '-' in objid:
                BAG_ids.append(objid)
                meshes.append(cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][2], vertices).clean())

        #what does this do?
        t = np.mean(main_mesh.points, axis=0)
        main_mesh.points -= t
        for mesh in meshes:
            mesh.points -= t

        # visualize candidate buildings + reference building
        # plot_meshes([main_mesh] + meshes)
        # plot = main_mesh.plot(scalars=cluster_meshes([main_mesh])[0])

        print(building_part)
        # plot the intersections between surfaces
        intersections = np.array

        merge_buildings.append(building_part)
        for nearby_building in range(len(meshes)):
            print(BAG_ids[nearby_building])
            intersection = intersect_surfaces([main_mesh, meshes[nearby_building]])
            if len(intersection) > 0:
                np.append(intersections,intersection)
                merge_buildings.append(BAG_ids[nearby_building])
                # show nearby building that has at lease one shared face
                # meshes[nearby_building].plot(scalars=cluster_meshes([meshes[nearby_building]])[0], text=f"Nearby building[{nearby_building}]")
                # show each of the shared faces individually
                # for i in range(len(intersection)):
                #     intersection[i].plot(text=f"Intersection[{i}]")
        print('done')

        # main_mesh.save("cluster.vtk")



