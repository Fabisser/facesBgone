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

# run hole filling 



# 2. run hole filling

echo "All done!";
	




