import numpy as np
import pyvista as pv
import pymesh
from helpers.geometry import plane_params, project_mesh, to_3d # module and defs
from scipy.spatial import distance_matrix
from sklearn.cluster import AgglomerativeClustering
import json
import cityjson # module
import os

def rpath(path):
    return os.path.expanduser(path)

float_formatter = "{:.3f}".format
np.set_printoptions(formatter={'float_kind':float_formatter})

filename = rpath("3dbag_v210908_fd2cee53_5907.json")

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
vertices = np.array(verts) # type = np.ndarray

# pick a CITY OBJECT to play with
obj1 = "NL.IMBAG.Pand.0503100000019229-0"
obj2 = "NL.IMBAG.Pand.0503100000019232-0"

building1 = cm["CityObjects"][obj1]
building2 = cm["CityObjects"][obj2]

"""Loading and plotting the first geometry as a pure pyvista (no triangulation done). """

"""What to_polydata is doing: 
        - it gets surface boundaries from the CityJSON geometry
        - it creates the faces this way
        - it uses the vertices and faces to call pv.PolyData and create the polydata MESH in pyvista
        - finally, it stores the semantics of each cell in a cell_data (dictionary) (mesh.cell_data)"""

 # Returns the triangulated polydata mesh from a CityJSON geometry.
trimesh1 = cityjson.to_triangulated_polydata(building1["geometry"][0], vertices).clean()
trimesh2 = cityjson.to_triangulated_polydata(building2["geometry"][0], vertices).clean()

# p = pv.Plotter()
# p.add_mesh(trimesh1, color="yellow")
# p.add_mesh(trimesh2, color="red")
# p.show()

"""Define a function to turn mesh into a pymesh from a pyvista PolyData."""
def to_pymesh(mesh):
    """Returns a pymesh from a pyvista PolyData"""
    v = mesh.points
    f = mesh.faces.reshape(-1, 4)[:, 1:]

    return pymesh.form_mesh(v, f)

"""Use the above function"""
m1 = to_pymesh(trimesh1)
m2 = to_pymesh(trimesh2)

wall = pymesh.boolean(m1, m2, operation="intersection", engine="igl")

to_pyvista(wall)