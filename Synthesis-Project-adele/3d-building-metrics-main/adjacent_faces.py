import json
import math
from concurrent.futures import ProcessPoolExecutor

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas
import pyvista as pv
import rtree.index
import scipy.spatial as ss
from pymeshfix import MeshFix
from tqdm import tqdm

import cityjson
import geometry
import shape_index as si

import numpy as np
import pyvista as pv
from helpers.geometry import plane_params, project_mesh, to_3d
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering

if custom_indices is None or len(custom_indices) > 0:
    voxel = pv.voxelize(tri_mesh, density=density_3d, check_surface=False)
    grid = voxel.cell_centers().points

    shared_area = 0

    closest_distance = 10000

    if len(neighbours) > 0:
        # Get neighbour meshes
        n_meshes = [cityjson.to_triangulated_polydata(geom, vertices).clean()
                    for geom in neighbours]

        for mesh in n_meshes:
            mesh.points -= t

        # Compute shared walls
        walls = np.hstack([geometry.intersect_surfaces([fixed, neighbour])
                           for neighbour in n_meshes])

        shared_area = sum([wall["area"][0] for wall in walls])

        # Find the closest distance
        for mesh in n_meshes:
            mesh.compute_implicit_distance(fixed, inplace=True)

            closest_distance = min(closest_distance, np.min(mesh["implicit_distance"]))

        closest_distance = max(closest_distance, 0)
    else:
        closest_distance = "NA"

    builder = StatValuesBuilder(values, custom_indices)

    builder.add_index("2d_grid_point_count", lambda: len(si.create_grid_2d(shape, density=density_2d)))
    builder.add_index("3d_grid_point_count", lambda: len(grid))

    builder.add_index("circularity_2d", lambda: si.circularity(shape))
    builder.add_index("hemisphericality_3d", lambda: si.hemisphericality(fixed))
    builder.add_index("convexity_2d", lambda: shape.area / shape.convex_hull.area)
    builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
    builder.add_index("convexity_3d", lambda: fixed.volume / ch_volume)
    builder.add_index("fractality_2d", lambda: si.fractality_2d(shape))
    builder.add_index("fractality_3d", lambda: si.fractality_3d(fixed))
    builder.add_index("rectangularity_2d", lambda: shape.area / shape.minimum_rotated_rectangle.area)
    builder.add_index("rectangularity_3d", lambda: fixed.volume / obb.volume)
    builder.add_index("squareness_2d", lambda: si.squareness(shape))
    builder.add_index("cubeness_3d", lambda: si.cubeness(fixed))
    builder.add_index("horizontal_elongation", lambda: si.elongation(S, L))
    builder.add_index("min_vertical_elongation", lambda: si.elongation(L, height_stats["Max"]))
    builder.add_index("max_vertical_elongation", lambda: si.elongation(S, height_stats["Max"]))
    builder.add_index("form_factor_3D", lambda: shape.area / math.pow(fixed.volume, 2 / 3))
    builder.add_index("equivalent_rectangularity_index_2d", lambda: si.equivalent_rectangular_index(shape))
    builder.add_index("equivalent_prism_index_3d", lambda: si.equivalent_prism_index(fixed, obb))
    builder.add_index("proximity_index_2d_", lambda: si.proximity_2d(shape, density=density_2d))
    builder.add_index("proximity_index_3d",
                      lambda: si.proximity_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("exchange_index_2d", lambda: si.exchange_2d(shape))
    builder.add_index("exchange_index_3d", lambda: si.exchange_3d(tri_mesh, density=density_3d))
    builder.add_index("spin_index_2d", lambda: si.spin_2d(shape, density=density_2d))
    builder.add_index("spin_index_3d",
                      lambda: si.spin_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("perimeter_index_2d", lambda: si.perimeter_index(shape))
    builder.add_index("circumference_index_3d", lambda: si.circumference_index_3d(tri_mesh))
    builder.add_index("depth_index_2d", lambda: si.depth_2d(shape, density=density_2d))
    builder.add_index("depth_index_3d", lambda: si.depth_3d(tri_mesh, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("girth_index_2d", lambda: si.girth_2d(shape))
    builder.add_index("girth_index_3d",
                      lambda: si.girth_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("dispersion_index_2d", lambda: si.dispersion_2d(shape, density=density_2d))
    builder.add_index("dispersion_index_3d",
                      lambda: si.dispersion_3d(tri_mesh, grid, density=density_3d) if len(grid) > 2 else "NA")
    builder.add_index("range_index_2d", lambda: si.range_2d(shape))
    builder.add_index("range_index_3d", lambda: si.range_3d(tri_mesh))
    builder.add_index("roughness_index_2d", lambda: si.roughness_index_2d(shape, density=density_2d))
    builder.add_index("roughness_index_3d",
                      lambda: si.roughness_index_3d(tri_mesh, grid, density_2d) if len(grid) > 2 else "NA")
    builder.add_index("shared_walls_area", lambda: shared_area)
    builder.add_index("closest_distance", lambda: closest_distance)

return obj, values

def cluster_meshes(meshes, threshold=0.1):
    """Clusters the faces of the given meshes"""

    n_meshes = len(meshes)

    # Compute the "absolute" plane params for every face of the two meshes
    planes = [face_planes(mesh) for mesh in meshes]
    mesh_ids = [[m for _ in range(meshes[m].n_cells)] for m in range(n_meshes)]

    # Find the common planes between the two faces
    all_planes = np.concatenate(planes)
    all_labels, n_clusters = cluster_faces(all_planes, threshold)
    areas = []

    labels = np.array_split(all_labels, [meshes[m].n_cells for m in range(n_meshes - 1)])

    return labels, n_clusters


def cluster_faces(data, threshold=0.1):
    """Clusters the given planes"""
    ndata = np.array(data)

    dm1 = distance_matrix(ndata, ndata)
    dm2 = distance_matrix(ndata, -ndata)

    dist_mat = np.minimum(dm1, dm2)

    clustering = AgglomerativeClustering(n_clusters=None,
                                         distance_threshold=threshold,
                                         affinity='precomputed',
                                         linkage='average').fit(dist_mat)

    return clustering.labels_, clustering.n_clusters_

def intersect_surfaces(meshes):
    """Return the intersection between the surfaces of multiple meshes"""

    def get_area_from_ring(areas, area, geom, normal, origin, subtract=False):
        pts = to_3d(geom.coords, normal, origin)
        common_mesh = pv.PolyData(pts, faces=[len(pts)] + list(range(len(pts))))
        if subtract:
            common_mesh["area"] = [-area]
        else:
            common_mesh["area"] = [area]
        areas.append(common_mesh)

    def get_area_from_polygon(areas, geom, normal, origin):
        # polygon with holes:
        if geom.boundary.type == 'MultiLineString':
            get_area_from_ring(areas, geom.area, geom.boundary[0], normal, origin)
            for sgeom in geom.boundary[1:]:
                get_area_from_ring(areas, 0, sgeom, normal, origin, subtract=True)
        # polygon without holes:
        elif geom.boundary.type == 'LineString':
            get_area_from_ring(areas, geom.area, geom.boundary, normal, origin)

    n_meshes = len(meshes)

    areas = []

    labels, n_clusters = cluster_meshes(meshes)

    for plane in range(n_clusters):
        # For every common plane, extract the faces that belong to it
        idxs = [[i for i, p in enumerate(labels[m]) if p == plane] for m in range(n_meshes)]

        if any([len(idx) == 0 for idx in idxs]):
            continue

        msurfaces = [mesh.extract_cells(idxs[i]).extract_surface() for i, mesh in enumerate(meshes)]

        # Set the normal and origin point for a plane to project the faces
        origin = msurfaces[0].clean().points[0]
        normal = msurfaces[0].face_normals[0]

        # Create the two 2D polygons by projecting the faces
        polys = [project_mesh(msurface, normal, origin) for msurface in msurfaces]

        # Intersect the 2D polygons
        inter = polys[0]
        for i in range(1, len(polys)):
            inter = inter.intersection(polys[i])

        if inter.area > 0.001:
            if inter.type == "MultiPolygon" or inter.type == "GeometryCollection":
                for geom in inter.geoms:
                    if geom.type != "Polygon":
                        continue
                    get_area_from_polygon(areas, geom, normal, origin)
            elif inter.type == "Polygon":
                get_area_from_polygon(areas, inter, normal, origin)
    return areas

input =