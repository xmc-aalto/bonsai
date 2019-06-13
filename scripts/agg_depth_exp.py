import re
import os
import pandas as pd
import sys

from glob import glob 
from collections import defaultdict


dataset = sys.argv[1]
path = '/scratch/cs/xml/bonsai-result/{}'.format(dataset)

def extract_floats(s):
    return re.findall('(\d+.\d+)', s)

rows_by_measure = defaultdict(list)

for path in glob(path + '/*/performance.txt'):
    segs = path.split('/')
    param_str = segs[-2]
    vals = re.findall('d(\d+)-rid(\d+)', param_str)[0]
    depth, rid = map(int, vals)

    with open(path, 'r') as f:
        for l in f:
            key = None
            if l.startswith('prec '):
                key = 'prec'
            elif l.startswith('nDCG '):
                key = 'nDCG'
            elif l.startswith('prec_wt '):
                key = 'prec_wt'
            elif l.startswith('nDCG_wt '):
                key = 'nDCG_wt'
            if key is not None:
                values = extract_floats(l)
                if len(values) != 5:
                    print('invalid output file for key {} of file {}'.format(key, path))
                    break
                rows_by_measure[key].append((depth, rid) + tuple(map(float, values)))

df_by_measure = {
    key: pd.DataFrame(rows, columns=['depth', 'rid', '1', '2', '3', '4', '5'])
    for key, rows in rows_by_measure.items()
}

for key, df in df_by_measure.items():
    print(key)
    print('-' * 10)
    res = df.groupby('depth')['1', '2', '3', '4', '5'].mean()
    print(res)
    print()
    output_path = '../outputs/{}/{}.csv'.format(dataset, key)
    if not os.path.exists('../outputs/{}'.format(dataset)):
        os.makedirs('../outputs/{}'.format(dataset))
    res.to_csv(output_path, sep=',')
