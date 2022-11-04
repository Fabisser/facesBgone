import numpy as np
import cityjson
def get_bbox(geom: dict, verts: np.ndarray) -> np.ndarray:
    pts = np.array(cityjson.get_points(geom, verts))
    return np.hstack([[np.min(pts[:, i]), np.max(pts[:, i])] for i in range(np.shape(pts)[1])])


def generator_function(cm: dict, verts: np.ndarray) -> tuple([int,tuple,str]):
    for i, objid in enumerate(cm["CityObjects"]):
        obj = cm["CityObjects"][objid]
        xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(obj["geometry"][0], verts)
        yield (i, (xmin, ymin, zmin, xmax, ymax, zmax), objid)