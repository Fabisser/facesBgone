## Logic
The code mainly follows the logic as written below:

**input**: cityjson file

**output**: cityjson file / .off file


* **Read**

	**read** from the `cityjson` file and store the geometries.
	- handle the repeatness in the original file.

	**read** from the `adjacency` file to get the adjacency list.
	- `adjacency` file can have multiple adjacent blocks or just one adjacent block, different adjacency file corresponds to different executing `mode`.

	**coordinates translation**
	- the original coordinates are quite large, for precision reasons the coordinates are translated.


* **Build**
	
	**build** **Polyhedron_3** from the stored geometries. 

	- extra care need to be taken since repeated vertices will cause problems when using `Polyhedron_incremental_builder`.
    - `self-intersection` will cause problems, thus **convex hull** is used if `Polyhedron_incremental_builder` yields errors.
    - about holes - handling holes can be a bit tricky, [CGAL](https://www.cgal.org/) has some related fucntions for that. Buildings with holes are stored in different manners in `cityjson` file and there are not many such buildings. Thus currently buidlings with holes are not properly handled (but there are some code usage [examples](https://github.com/zfengyan/geoCFD/blob/v1/src/Polyhedron.hpp#L189) in the file -> [Polyhedron.hpp](https://github.com/zfengyan/geoCFD/blob/v1/src/Polyhedron.hpp).
  
	**build** **Nef_polyhedron_3** from the **Polyhedron_3**.

	- only if **Polyhedron_3** is closed
    - **Nef_polyhedron_3** will complain when the points of surfaces are not **planar**, which will lead to a **invalid Nef_polyhedron_3**, thus `triangulation` is required (the surfaces of **Polyhedron_3** will firstly be triangulated and then converted to **Nef_polyhedron_3**).


* **Minkowski sum**

	- expand each **Nef_polyhedron_3** with a cube, the side length of the cube can be decided by user (`0.01` by default).
    - **minkowski sum** sometimes can be a bit picky, it will probably complain about complicated buildings. The buildings causing CGAL assertions are handled (replaced by its convex hull). However, it should be noted that some buildings may cause segfault, which is hard to handle. So the code is not that robust.
    - robust - use TetGen to decompose the buildings into convex parts first, expand the convex parts and then merge them to get the expanded **Nef_polyhedron_3** [ possible solultion, not implemented yet ]


* **Post processing**

	- merge all the expanded **Nef_polyhedron_3** into one **big Nef_polyhedron_3**.
	- extract geometries from the **big Nef_polyhedron_3**.
    - remeshing - since triangulation is applied before building **Nef_polyhedron_3**, some skinny triangles can result in self-intersection faces. [Remeshing](https://doc.cgal.org/latest/Polygon_mesh_processing/index.html) can more or less refactors the triangulated mesh, but in the practice good result requires very long execution time and high computational resources. This [function](https://github.com/zfengyan/geoCFD/blob/v1/src/Polyhedron.hpp#L745) can be activated if user wants to (a super computer will help).
    - erosion - erosion can be used to make expanded **Nef_polyhedron_3** get back to its original shape, but usually erosion will introduce more irregular faces at the same time, thus this function is not used. The example is [here](https://github.com/zfengyan/geoCFD/blob/v1/src/Polyhedron.hpp#L728).

### JsonHandler.hpp
Responsible for taking care of the input `.cityjson` file and store the necessary information(i.e., `Solid`, `Shell`, `Face`, `Vertices` of one building(part)).

### JsonWriter.hpp
Responsible for writing the result to a `.cityjson` file so that the result can be visualised in [ninja](https://ninja.cityjson.org/). It should be noted that users can choose to export the **exterior** or **interior** of the result buidling.

### Polyhedron.hpp
Responsible for ->

- **(a)** taking care of `repeatness` in the dataset - `Polyhedron builder` doesn't like repeated vertices.
    
- **(b)** passing the **correct** geometry information of a building(part) to `Polyhedron builder`.
    
- **(c)** building the `CGAL::Polyhedron_3` for each building(part).
    
- **(d)** converting the `CGAL::Polyhedron_3` to `CGAL::Nef_polyhedron_3` if `CGAL::Polyhedron_3` is `closed` otherwise use the `convex hull` instead.

- **(e)** unioning all the `CGAL::Nef_polyhedron_3` into one big `CGAL::Nef_polyhedron_3` via CSG operations.

  - **(e1)** applying the `minkowski sum` first on each `CGAL::Nef_polyhedron_3` and then performing the union operation(to eliminate the small gaps and then to eliminate the internal faces).
        
  - **(e2)** getting the `convex hull` of the big `CGAL::Nef_polyhedron_3` to eliminate the internal faces.
    
- **(f)** extracting the geometry information(i.e. `vertices`, `faces`) of the big `CGAL::Nef_polyhedron_3` via a visitor pattern.
    
- **(g)** processing the obtained geometry information -> to write the result to `json` file correctly, the obtained geometry information needs to be further processed.

### MultiThread.hpp
Responsible for performing multi-threading processing.

### main.cpp

The entry point of the whole program.

## Attention
1. **vertices repeatness**

	Using `CGAL::Polyhedron_3` can be tricky, since `Polyhedron_builder` doesn't like repeated vertices, and even if it workswith repeatness, the created `Polyhedron_3` is NOT closed(and thus can not be converted to `Nef_polyhedron`).Thus extra care needs to be taken when creating `Polyhedron_3`.

2. **geometry error**

	In `lod 2.2` geometries of buildings can get complicated, in our test there can be some geometry errors in the original dataset, for example:
	```console
	build nef for building: NL.IMBAG.Pand.0503100000018412-0

	CGAL::Polyhedron_incremental_builder_3<HDS>::
	lookup_halfedge(): input error: facet 29 has a self intersection at vertex 79.
	polyhedron closed? 0
	```
	A manual fix can be possible towards certain building set but not general, if we take the **universality** and **automation** into consideration, using `convex hull` to replace the corresponding building seems to be a good choice, yet this approach may lead to the compromisation of the original building shapes.
	
	There are also other issues in `lod 2.2`, see [robust](https://github.com/SEUZFY/geoCFD/tree/master#robust) section for details.
	
## Robust

One important aspect is that `CGAL::Nef_polyhedron_3` will complain when the points of surfaces are not planar, which will lead to a `invalid` `Nef_polyhedron`.

A possible solution is demonstrated as below (it has been tested on building set 1):
```cpp
typedef CGAL::Simple_cartesian<float> Kernel;
typedef CGAL::Polyhedron_3<Kernel> Polyhedron_3;

Polyhedron_3 polyhedron = load_my_polyhedron();
CGAL::Polygon_mesh_processing::triangulate_faces(polyhedron);
```
A triangulation process of the surfaces of a polyhedron can make points planar since there's always a plane that passes between any three points, no matter their locations but this is not guaranteed when having four or more points, which is pretty common in `lod 2.2`, among 23 `Nef polyhedra` from 23 buildings, 22 of them are
`invalid` and only 1 is `valid`.

It should however be noted that, triangulating process will modify the original geometry a bit (tiny change but for now not sure how to visualise / quantify the possible influence) and also have time cost.

Also we need to take the `geometry validity` of `CGAL::Polyhedron_3` into consideration. The `is_valid()` function only checks for `combinatorial validity` (for example in every half-edge should have an opposite oriented twin) but does not check the geometry correctness. In order to check it we can use `CGAL::Polygon_mesh_processing::do_intersect` to check for example if there are any intersections (need to verify and test).

Why do we need to check `geometry validity`? Because this could break the corresponding Nef polyhedron, for example, the corresponding Nef polyhedron is not valid.

possible solutions: allow users to switch on/off different robust check functions by defining different macros, for example:
```cpp
#define _POLYHEDRON_3_GEOMETRY_CHECK_
#define _POLYHEDRON_3_COMBINATORIAL_CHECK_
...
```

issues related minkowski sum and irregular building:

[CGAL Minkowski sum assertion when performing union operation of Nef_polyhedron_3](https://github.com/CGAL/cgal/issues/6973)
