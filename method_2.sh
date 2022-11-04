#! /bin/bash

# flag definition
while getopts ":hi:c:mink:h:e:" option; do
   case $option in
	h) # display Help
		Help
		exit;;
	i) inter_area=$OPTARG;;
	c) cluster_meshes=$OPTARG;;
	mink) minkowski=$OPTARG;;
	h) max_hole=$OPTARG;;
	e) max_edges=$OPTARG;;
   esac
done

# run adjacency and NEF

# 1. adjacency 
cd /mnt/c/Users/Panagiotior/Desktop/Test_Jupyter/Synthesis-Project/3d-building-metrics/Symdiff_Fab/
python3 main.py "${inter_area:-$default_inter_area}" "${cluster_meshes:-$default_cluster_meshes}"

# 2. NEF 
# # run nef polyhedra => need to modify cpp to take cluster.txt as second argument
# cd /home/atherias/synthesis-NEF/build/
echo "minkowski value is : $minkowski"
cd /mnt/c/Users/Panagiotior/Desktop/Test_Jupyter/Synthesis-Project/3d-building-metrics/Symdiff_Fab/synthesis-NEF/build/
chmod a+x geocfd
adjacencies = "/mnt/c/Users/Panagiotior/Desktop/Test_Jupyter/Synthesis-Project/3d-building-metrics/Symdiff_Fab/data/adjacencies.txt"
./geocfd $filename $adjacencies "${minkowski:-$default_minkowski}" "${max_hole:-$default_max_hole}" "${max_edges:-$default_max_edges}"

echo "All done!";
	


