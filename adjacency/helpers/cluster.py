import numpy as np
import scipy
from sklearn.cluster import AgglomerativeClustering
import pyvista as pv
def plane_params(normal: np.ndarray, origin: np.ndarray, rounding=2) -> np.ndarray:
    """Returns the params (a, b, c, d) of the plane equation for the given
    normal and origin point.
    """
    a, b, c = np.round_(normal, 3)
    x0, y0, z0 = origin
    
    d = -(a * x0 + b * y0 + c * z0)
    
    if rounding >= 0:
        d = round(d, rounding)
    
    return np.array([a, b, c, d])

def face_planes(mesh: pv. PolyData) -> np.ndarray:
    return [plane_params(mesh.face_normals[i], mesh.cell_points(i)[0]) for i in range(mesh.n_cells)]


def cluster_faces(data: np.ndarray, threshold: int) -> tuple([np.ndarray, int]):
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
    
    # Find the common planes between the two meshes
    # array of parameters for all planes in both meshes (combined)
    all_planes = np.concatenate(planes)

    all_labels, n_clusters = cluster_faces(all_planes, threshold)
    

    # list of arrays (one for each mesh) indicating cluster labels for each face
    labels = np.array_split(all_labels, [meshes[m].n_cells for m in range(n_meshes - 1)])
    
    return labels, n_clusters