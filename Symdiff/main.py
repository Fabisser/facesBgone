from email import generator
import json
import numpy as np
import pyvista as pv
import math
import os
from helpers.bbox import *
from helpers.geometry import *
from helpers.cluster import *
from shapely.geometry import MultiPolygon, Polygon, LineString
from symdiff import *
import scipy
from sklearn.cluster import AgglomerativeClustering
import rtree.index
import cityjson
import matplotlib.pyplot as plt
import time

print("go time")
start = time.time()
# Implementation

# Load cityjson
float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

filename = "data/tile.json"

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
# create a dictionary with the index as key and the coords as values
verts_dict = {}
for i, vert in enumerate(vertices):
    verts_dict[i] = [vert[0], vert[1], vert[2]]


# what is this?
p = rtree.index.Property()
p.dimension = 3
r = rtree.index.Index(generator_function(cm, vertices), properties=p)


# MULTIPLE BUILDING VERSION
clustered_buildings = {}
# for building_part in cm['CityObjects']:
#     if '-' in building_part:   
#         t += np.mean(cityjson.to_triangulated_polydata(cm["CityObjects"][building_part]["geometry"][2], vertices).clean().points, axis=0)

# t = 2*t/len(cm["CityObjects"])
finalmesh = pv.PolyData()
finalmesh = finalmesh.triangulate()
passed_building = []
for building_part in cm['CityObjects']:           
                
    if '-' in building_part:    
        passed_building.append(building_part)
        
        """ ACCESSing CityJson data model *** NOT TRIANGULATED """ # --> no need to access 
        xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(cm["CityObjects"][building_part]["geometry"][0], verts)
        objids = [n.object for n in r.intersection((xmin, ymin, zmin, xmax, ymax, zmax), objects=True) if n.object != building_part]
        # this is the name of each building 
        # POLYDATA object (TRIANGULATED)
        main_mesh = cityjson.to_triangulated_polydata(cm["CityObjects"][building_part]["geometry"][2], vertices).clean()
        numf_b1 = main_mesh.n_cells
        
                        
        # print(main_mesh.points)
        
        # meshes = [cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][2], vertices).clean() for objid in objids if '-' in objid]
        meshes = []
        meshes_id = []
        for objid in objids:
            if '-' in objid: # and objid in check_list:               
                meshes.append(cityjson.to_triangulated_polydata(cm["CityObjects"][objid]["geometry"][2], vertices).clean())
                meshes_id.append(objid)

        # Make origin of the building mesh at the center (mean) of the points
        t = np.mean(main_mesh.points, axis=0)
        main_mesh.points -= t
        for mesh in meshes:
            mesh.points -= t
            
        for nearby_building in range(len(meshes)):
            all_idxs, symdif = symmetric_difference([main_mesh, meshes[nearby_building]])
            if len(symdif) != 0:
                
                #Create combined symmetric difference    
                symdif_mesh = symdif[0]
                for i in range(1, len(symdif)):
                    symdif_mesh = symdif_mesh + symdif[i]

                idx = sum(all_idxs, [])
                main_mesh = main_mesh.remove_cells(idx)             
 
                if not (meshes_id[nearby_building] in passed_building): #We only want to add symdiff for each building once
                    symdif_mesh.points += t
                    finalmesh += symdif_mesh
        main_mesh.points += t
        finalmesh += main_mesh
finalmesh = finalmesh.clean()
# plotter = pv.Plotter()
# ## add meshes
# actor =  plotter.add_mesh(finalmesh, opacity = 0.3, color = "red", show_edges = True)
            
# plotter.show()   

finalmesh.save("data/Outputtile.stl")
end = time.time()

print("Time consumed in working: ",end - start)