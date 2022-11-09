#! /bin/bash
Help()
{
   # Display Help
   echo ""
   echo "facesBgone removes faces between adjacent buildings from a CityJSON dataset in LoD 2.2 (e.g. 3D BAG)."
   echo "For more information, refer to README.md."
   echo
   echo "Syntax: sh run.sh [h|m|i|c|d|e|p]"
   echo
   echo "General parameters:"
   echo "	h	Print this Help."
   echo "	m	specify method 1 (hole filling) or 2 (nef polyhedra)."
   echo "		If no method is specified, both methods will be run."
   echo 
   echo "Adjacency parameters:"
   echo "	i	threshold to define intersection between adjacent faces."
   echo "		default i = 0.1"
   echo "	c	threshold to use in agglomerative clustering."
   echo "		default c = 0.1"
   echo
   echo "Hole filling parameters (both have to be >0 to be considered):"
   echo "	d	maximum diameter of holes to be filled."
   echo "		default hole = -1"
   echo "	e	maximum number of edges of holes to be filled."
   echo "		default e = -1"
   echo
   echo "Nef Polyhedra parameters:"
   echo "	p	value used for minkowski sum."
   echo "		default mink = 0.003"
   
}

# flag definition
while getopts ":hm:i:c:p:d:e:" option; do
   case $option in
	h) # display Help
		Help
		exit;;
	m) method=$OPTARG;;
	i) inter_area=$OPTARG;;
	c) cluster_meshes=$OPTARG;;
	p) minkowski=$OPTARG;;
	d) max_hole=$OPTARG;;
	e) max_edges=$OPTARG;;
   esac
done

curdir="$(pwd)"

# get inputs from user
echo "Enter CityJson File Name: ";
read filename;

if [ -z $filename ];
then
	echo "Please specify file name next time :)"
	exit 1
else
	echo "-----Input file name: $filename------";
fi

# 1. adjacency

cd ./adjacency
python3 main.py $filename "${inter_area:-$default_inter_area}" "${cluster_meshes:-$default_cluster_meshes}"

if [ -z $minkowski]
then
	minkowski=0.01
fi

if [ -z $method ];
then
    echo "Proceding with both methods..."
	
	# run hole_filling => need to modify cpp to take the correct arguments
	cd ../hole-filling/build/
	chmod a+x hole-filling
	./hole-filling $curdir $max_hole $max_edges



	cd ../../nef/build/
	chmod a+x geoCFD
	./geoCFD $filename $curdir/data $minkowski


else
    echo "Your method of choice is: $method"
	if [ "$method" = 1 ]; 
	then
		echo "Proceeding with Hole Filling Method"
		cd ../hole-filling/build/
		chmod a+x hole-filling
		./hole-filling $curdir $max_hole $max_edges
	fi

	if [ "$method" = 2 ]; 
	then
		echo "Proceeding with NEF Method"
		cd ../nef/build/
		chmod a+x geoCFD
		./geoCFD $filename $curdir/data $minkowski
	fi
fi 
