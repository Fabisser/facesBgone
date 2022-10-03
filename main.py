from pycode import *
import build.convex_hull

print("Calculating Adjacancy")
path = "pycode/blocks_of_adjacent_buildings_subset_5907.json"
list = common_walls(path)

build.convex_hull.calculate(list)