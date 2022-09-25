from pycode import *
import build.convex_hull
from build.convex_hull import *
path = "pycode/blocks_of_adjacent_buildings_subset_5907.json"
list = common_walls(path)

print(list)

build.convex_hull.calculate(list)