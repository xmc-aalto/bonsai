import json
import numpy as np
import networkx as nx


def parse_model(tree_path, bonsai_format=True):
    """return a tree in networkx.DiGraph"""
    READ_IS_LEAF = 'is_leaf'
    READ_CHILDREN = 'children'
    READ_DEPTH = 'depth'
    READ_LABELS = 'labels'
    READ_FEATURE_DIM = 'feature_dim'
    READ_FEATURE_COL = 'feature_col'
    
    with open(tree_path, 'r') as f:
        g = nx.DiGraph()
        # skip |X| and |Y|
        for l in range(2):
            f.readline()
        num_nodes = int(f.readline().strip())
        g.add_nodes_from(np.arange(num_nodes))

        state = READ_IS_LEAF
        current_node = -1
        for l in f:
            l = l.strip()
            if state == READ_IS_LEAF:
                inner_column_index = -1
                current_node += 1
                is_leaf = bool(int(l))
                g.node[current_node]['is_leaf'] = is_leaf
                
                print('current node', current_node)
                print('is leaf', is_leaf)
                if is_leaf and bonsai_format:
                    state = READ_DEPTH
                else:
                    state = READ_CHILDREN
            elif state == READ_CHILDREN:
                children = map(int, l.split())
                for child in children:
                    if not g.node[current_node]['is_leaf']:
                        g.add_edge(current_node, child)
                state = READ_DEPTH
                print('children', list(g[current_node].keys()))
            elif state == READ_DEPTH:
                g.node[current_node]['depth'] = int(l)
                print('depth', int(l))
                state = READ_LABELS
            elif state == READ_LABELS:
                segs = l.split()
                num_labels, labels = int(segs[0]), list(map(int, segs[1:]))
                assert len(labels) == num_labels
                g.node[current_node]['labels'] = labels
                g.node[current_node]['num-labels'] = len(labels)
                print('#labels', len(labels))
                state = READ_FEATURE_DIM
            elif state == READ_FEATURE_DIM:
                n_feature_cols, _ = map(int, l.split())
                state = READ_FEATURE_COL
                print('n_feature_cols', n_feature_cols)
            elif state == READ_FEATURE_COL:
                # skip feature for now
                inner_column_index += 1
                if inner_column_index == (n_feature_cols - 1):
                    state = READ_IS_LEAF
                    print('to new node')
                else:
                    state = READ_FEATURE_COL
                    print('is feature col, skip')
    assert num_nodes == g.number_of_nodes(), '{} != {}'.format(
        num_nodes, g.number_of_nodes())
    assert g.number_of_edges() == (num_nodes - 1), '{} != {}'.format(
        g.number_of_edges(), (num_nodes - 1))
    assert len(list(nx.weakly_connected_components(g))) == 1
    
    return g


def gen_nested_tree(g):
    def aux(n):
        """return a list of child nodes"""
        # print('at node ', n, g.node[n])
        if g.node[n]['is_leaf']:
            return {'name': n, 'value': g.node[n]['num-labels']}
        else:
            return {
                'name': n,
                'children': [aux(c) for c in g[n].keys()]
                }
    return aux(0)

if True:
    input_path = '/home/xiaoh1/code/parabel-v2-old/sandbox/results/eurlex/model/0.tree'
    output_path = '../outputs/tree_json/parabel.json'
else:
    input_path = '/scratch/cs/xml/bonsai-model/eurlex/d3-rid1/0.tree'
    output_path = '../outputs/tree_json/eurlex-d3.json'

print('input ', input_path)    

g = parse_model(input_path, bonsai_format=False)
d = gen_nested_tree(g)

print('output to ', output_path)

json.dump(d, open(output_path, 'w'))
