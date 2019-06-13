import math
import sys

depths = (2, 3, 4, 5, 6)
dataset = sys.argv[1]

def get_fanout(L, d):
    return math.ceil(L**(1/d))

ds2L = {
    'eurlex': 3993,
    'wiki10': 30938,
    'wikiLSHTC': 325056,
    'WikipediaLarge-500K': 501070
}

L = ds2L[dataset]

fanouts = [get_fanout(L, d) for d in depths]

run_ids = list(range(1, 11))

for fanout, depth in zip(fanouts, depths):
    for run_id in run_ids:
        print("./depth_exp.sh {dataset} {fanout} {max_depth} {run_id}".format(
            dataset=dataset,
            fanout=fanout,
            max_depth=depth,
            run_id=run_id
        ))

