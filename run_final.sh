#! /bin/bash
Help()
{
   # Display Help
   echo "Add description of the script functions here."
   echo
   echo "Syntax: scriptTemplate [-g|h|v|V]"
   echo "options:"
   echo "g     Print the GPL license notification."
   echo "h     Print this Help."
   echo "v     Verbose mode."
   echo "V     Print software version and exit."
   echo
}

# add option to choose method 1 or 2


# flag definition
while getopts ":hm:i:c:mink:h:e:" option; do
   case $option in
	# h) # display Help
	# 	Help
	# 	exit;;
	m) method=$OPTARG;;
	i) inter_area=$OPTARG;;
	c) cluster_meshes=$OPTARG;;
	mink) minkowski=$OPTARG;;
	h) max_hole=$OPTARG;;
	e) max_edges=$OPTARG;;
   esac
done


# get inputs from user
echo "Enter CityJson File Name: ";
read filename;
echo "-----Input file name: $filename------";

echo "-----User-defined Adjacency parameters-----";
echo "Intersection area threshold: $inter_area";
echo "Cluster meshes threshold: $cluster_meshes";

echo "-----User-defined Hole Filling parameters-----";
echo "Max hole value: $max_hole";
echo "Max number of hole edges: $max_edges";

echo "-----User-defined Nef Polyhedron Method parameters-----";
echo "Minkowski value: $minkowski";

		
		
# 1. adjacency
echo "Running adjacency"
cd ./adjacency/
python3 main.py $filename "${inter_area:-$default_inter_area}" "${cluster_meshes:-$default_cluster_meshes}"


if [ -z $method ];
then
    echo "Doing both methods"
	
	# run hole_filling => need to modify cpp to take the correct arguments
	cd ./../hole_filling/build/
	# chmod a+x hole-filling
	# ./hole-filling $filename
	
	# run nef polyhedra => need to modify cpp to take cluster.txt as second argument
	# cd ./../../data/
	# adjacencies =  "adjacencies.txt"
	cd ./../../nef/build/
	chmod a+x geocfd
	./geocfd $filename ./../../data/adjacencies.txt $minkowski	
	
else
    echo "Method : $method"
fi


echo "Your method of choice is: $option"



# OPTION 1 
if [ "$option" = 1 ]; then
echo "Proceeding with Hole Filling Method"
	
	
	sh ./method_1.sh -i "${inter_area:-$default_inter_area}" -c "${cluster_meshes:-$default_cluster_meshes}"
	
fi

# OPTION 2
if [ "$option" = 2 ]; then
echo "Proceeding with NEF Method"

	echo "set your thresholds or press enter for default"
	
	echo "intersection area (press enter for default):"
	read inter_area;
	echo "cluster meshes (press enter for default):"
	read cluster_meshes;
	echo "minkowski (press enter for default):"
	read minkowski;
	
	sh ./method_2.sh "${inter_area:-$default_inter_area}" -c "${cluster_meshes:-$default_cluster_meshes}" -mink "${minkowski:-$default_minkowski}"

fi

