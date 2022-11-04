import sys
import json
import numpy as np
import pyvista as pv
from helpers.bbox import *
from helpers.geometry import *
from helpers.cluster import *
from symdiff import *
import rtree.index
import cityjson
import networkx
import tqdm
from networkx.algorithms.components.connected import connected_components


# Implementation

# beginning for functions for adjacency
def to_graph(l):
    G = networkx.Graph() # creating a graph
    for part in l:
        # each sublist is a bunch of nodes
        G.add_nodes_from(part)
        # it also imlies a number of edges:
        G.add_edges_from(to_edges(part))
    return G

def to_edges(l):
    """
        treat `l` as a Graph and returns it's edges
        to_edges(['a','b','c','d']) -> [(a,b), (b,c),(c,d)]
    """
    it = iter(l)
    last = next(it)

    for current in it:
        yield last, current
        last = current
# end of functions for adjacency

def main(argv):
    # Load cityjson
    float_formatter = "{:.3f}".format
    np.set_printoptions(formatter={'float_kind':float_formatter})

    filename = "data/" + argv[0]

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

    p = rtree.index.Property()
    p.dimension = 3
    r = rtree.index.Index(generator_function(cm, vertices), properties=p)

    # building id as key, list of buildings in the same block as value

    clustered_buildings_list = []
    cluster_id = 0

    # put all cityjson objects into a dictionary of meshes
    # building id as key, mesh as value
    mesh_dict = {}

    # create dict to track adjacency tests
    adjacency_tests = {}

    #Finalmesh outputs the mesh without internal faces. The seperate buildings are however not connected
    finalmesh = pv.PolyData()

    #buildings with no adjacancies
    single_buildings = []



    # IMPLEMENTATION
    # save all buildings' geometry in a mesh_dict (except of the invalid buildings)
    for building_part in cm["CityObjects"]:
        # filter out the invalid buildings
        if '-' in building_part:
            mesh_dict[building_part] = cityjson.to_triangulated_polydata(cm["CityObjects"][building_part]["geometry"][2],
                                                                        vertices).clean()
            adjacency_tests[building_part] = []


    for building, main_mesh in tqdm.tqdm(mesh_dict.items()):


        building_and_neigbs = []
        building_and_neigbs.append(building)


        # Make origin of the building mesh at the center (mean) of the points of the main mesh
        t = np.mean(mesh_dict[building].points, axis=0)
        # remove t from the reference building to set its mean as the origin of the CRS
        mesh_dict[building].points -= t


        # get bounding box of reference building, find the objects that intersect its bbox
        xmin, xmax, ymin, ymax, zmin, zmax = get_bbox(cm["CityObjects"][building]["geometry"][0], verts)
        objids = [n.object for n in r.intersection((xmin, ymin, zmin, xmax, ymax, zmax), objects=True) if
                n.object != building]


        if len(objids) == 0:
            single_buildings.append([building])

        for objid in objids:
            if '-' in objid:

                # first, check if objid is a key
                if objid in adjacency_tests.keys():
                    # then check if the len > 0 --> otherwise it gives KEYERROR in case the key doesn't exist

                    # if the key has values
                    if len(adjacency_tests[objid]) > 0:
                        if building in adjacency_tests[objid]:
                            continue
                        else:
                            # add to the dict all potential neighbs
                            adjacency_tests[building] = adjacency_tests[building] + [objid]
                            # test adjacency between building and objid
                            # remove from the neighbouring buildings the t, to keep the same relative distance
                            mesh_dict[objid].points -= t
                            all_idxs, symdif = symmetric_difference([mesh_dict[building], mesh_dict[objid]])

                            if len(symdif) != 0:
                                # add to the adjacencies
                                building_and_neigbs.append(objid)
                                # Create combined symmetric difference
                                symdif_mesh = symdif[0]
                                for i in range(1, len(symdif)):
                                    symdif_mesh = symdif_mesh + symdif[i]
                                # restore coordinates for symdif
                                symdif_mesh.points += t
                                finalmesh += symdif_mesh

                                # remove intersection faces from reference building
                                idx_0 = sum(all_idxs[0], [])
                                idx_1 = sum(all_idxs[1], [])
                                
                                mesh_dict[building] = mesh_dict[building].remove_cells(idx_0) #Remove internal faces from current building

                                mesh_dict[objid] = mesh_dict[objid].remove_cells(idx_1) #Remove internal faces from adjacent building


                            # restore coordinates for mesh_dict of the neighbouring building
                            mesh_dict[objid].points += t


                    # else if the key has no values
                    else:

                        adjacency_tests[building] = adjacency_tests[building] + [objid]
                        # test adjacency between building and objid
                        # remove from the neighbouring buildings the t, to keep the same relative distance
                        mesh_dict[objid].points -= t
                        all_idxs, symdif = symmetric_difference([mesh_dict[building], mesh_dict[objid]])
                        
                        if len(symdif) != 0:
                            
                            # add to the adjacencies
                            building_and_neigbs.append(objid)
                            
                            # Create combined symmetric difference
                            symdif_mesh = symdif[0]
                            for i in range(1, len(symdif)):
                                symdif_mesh = symdif_mesh + symdif[i]
                                
                            # restore coordinates for symdif
                            symdif_mesh.points += t
                            finalmesh += symdif_mesh

                            # remove intersection faces
                            idx_0 = sum(all_idxs[0],[])
                            idx_1 = sum(all_idxs[1],[])

                            mesh_dict[building] = mesh_dict[building].remove_cells(idx_0) #Remove internal faces from current building

                            mesh_dict[objid] = mesh_dict[objid].remove_cells(idx_1) #Remove internal faces from adjacent building

                    # restore coordinates for mesh_dict of the neighbouring building
                    mesh_dict[objid].points += t
        # for loop finished

        clustered_buildings_list.append(building_and_neigbs)
        # restore coordinates for mesh_dict of the reference building
        mesh_dict[building].points += t

    #Adjancancies are calculated
    G = to_graph(clustered_buildings_list)
    Glist = connected_components(G)
    list_of_adjacencies = list(Glist)


    # Fengyan Output --> write in a txt
    f = open("data/adjacencies_final.txt", "w")
    for l in list_of_adjacencies:
        for el in l:
            f.write(el + "\n")
        f.write("\n")
    f.close()

    # add in the final mesh all the meshes from mesh_dict (all buildings with removed internal faces)
    for num, mesh in enumerate(mesh_dict):
        finalmesh += mesh_dict[mesh]

    # output in stl
    finalmesh = finalmesh.clean()
    finalmesh.save("data/Outputtile_mesh.stl")

if __name__ == "__main__":
    print("Calculating Adjacancies..")
    main(sys.argv[1:])
    print("Adjacancies saved to ...")