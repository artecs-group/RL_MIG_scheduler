import sys
sys.path.append('..')
from basic_agent.utils import partition_map

for part, val in partition_map.items():
    i = 0
    print(', "instances" : [', end="")
    for instance in val["sizes"]:
        for _ in range(instance):
            print(f"{i},", end="")
        i += 1
    print("]")
