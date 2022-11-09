# facesBgone 
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Linux](https://svgshare.com/i/Zhy.svg)](https://svgshare.com/i/Zhy.svg)
[![Generic badge](https://img.shields.io/badge/updated-2022-<COLOR>.svg)](https://github.com/Fabisser/Synthesis-Project)

## About the project

This application removes faces between adjacent building from a [CityJSON](http://www.cityjson.org/ "CityJSON") dataset in LoD 2.2 (e.g. 3D BAG).
It was developed as a MSc Geomatics Synthesis Project in November 2022.

**Authors:**<br>
- Ioanna Panagiotidou
- Chrysanthi Papadimitriou
- Eleni Theodoridou
- Adele Therias
- Fabian Visser
- Fengyan Zhang

**Academic Supervisors:**<br>
- Dr. Clara García-Sánchez <br>
*Assistant Professor | 3D geoinformation group | TU Delft*
- Ivan Pađen <br>
*PhD Candidate | 3D geoinformation group | TU Delft*

**Industry Partner:**<br>
- Ignacio Gonzalez-Martino <br>
*Senior Manager for Aerospace | Dassault Systèmes*

Special thanks to Stelios Vitalis, Dr. Ken Arroyo Ohori, Dr. Liangliang Nan and Dr. Hugo Ledoux for their support during this project.

## Methods
1. Python script identifies blocks of adjacent buildings and outputs a list of blocks containing building ids as a .txt
2. The adjacency information is used to run two different methods (developed in C++) for reconstructing the geometry without shared faces:
   - **Hole filling**: uses symmetric difference between adjacent buildings to extend and close each block as one mesh.
   - **Nef Polyhedra**: constructs and merges nef polyhedra from adjacent buildings in a block, applying Minkowski sum to remove gaps between buildings
   

https://user-images.githubusercontent.com/79523968/200867898-803496d9-1459-4ba1-8b6e-f1b55308a27a.mov


## Documentation
A detailed description of the methods and parameters can be found in the final report [here](https://www.tudelft.nl/en/education/programmes/masters/geomatics/msc-geomatics/programme/synthesis-project/)

## Folder structure
- **facesBgone-main**: project folder where Bash files are saved.
  - **data**: contains all input and output files.
  - **adjacency**: contains all python scripts.
  - **hole_filling**: contains executable and scripts for Hole Filling method (c++)
  - **nef**: contains executable and scripts for Nef Polyhedra method (c++)
  
## Set up
This application uses C++ and Python, and must be run via a Linux/Unix command line.
1. Install [CGAL](https://www.cgal.org/ "CGAL") > v.5.5.1. 
   - To install on Linux or Windows (WSL) use `sudo apt-get install libcgal-dev`
   - To install on MacOS use `brew install cgal`
2. Run `chmod u+x setup.sh` then `sh setup.sh`: this installs the Python requirements. Python packages can also be installed with `pip install -r requirements`
3. Run `chmod u+x run.sh`

For MacOS just run `sh setup.sh`

## Usage
1. Prepare CityJSON dataset and place in *project_face_removal/adjacency/data/*
2. Open (WSL) command line from *project_face_removal* folder.
3. Run the bash file, optionally specifying method to use and/or parameters. If no parameters are entered, default values will be used and both methods will run.
   - Syntax: `sh run.sh [h|i|c|p|h|e|d]`
   - e.g.`sh run.sh` or `sh run.sh -mink 0.002 `
4. When prompted, enter file name of CityJSON dataset.
   - e.g. *buildings.city.json*

## Optional: User-defined parameters
You can also view all optional user-defined parameters by running `sh run.sh -h`.

**method**`-m`: specify method 1 (hole filling) or 2 (nef polyhedra). If no method is specified, both methods will be run.<br>

### Adjacency parameters<br>
**inter_area**`-i`: threshold to define intersection between adjacent faces. default i = 0.1<br>
**cluster_meshes**`-c`: threshold to use in agglomerative clustering. default c = 0.1<br>

### Hole filling parameters<br>
Both have to be positive to be considered
**max_hole**`-d`: maximum diameter of holes to be filled. default hole = -1<br>
**max_edges**`-e`: maximum number of edges of holes to be filled. default e = -1<br>

### Nef Polyhedra parameters<br>
**minkowski**`-p`: value used for minkowski sum. default mink = 0.003<br>

## References
- Adjacency script adapted from [3D Building Metrics](https://github.com/tudelft3d/3d-building-metrics/). <br>
Anna Labetski, Stelios Vitalis, Filip Biljecki, Ken Arroyo Ohori & Jantien Stoter (2022) 3D building metrics for urban morphology, International Journal of Geographical Information Science, DOI: 10.1080/13658816.2022.2103818
