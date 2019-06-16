"""
    Split data into train (0.8), dev (0.1), test (0.1)
"""
import numpy as np


data_list = [
    # twitter
    'twitter',
    # amazon
    'amazon',
    # yelp hotel
    'yelp_hotel',
    # yelp restaurant
    'yelp_rest',
]

def split(datap):
    print('Working on:', datap)
    with open(datap + '/' + datap + '.tsv') as dfile: 
        cols = dfile.readline()
        lines = dfile.readlines()
        indices = list(range(len(lines)))

        np.random.shuffle(indices)
        
        with open(datap + '/' + datap + '.train', 'w') as wfile:
            wfile.write(cols)
            for idx in indices[:int(0.8*len(lines))]:
                wfile.write(lines[idx])
        with open(datap + '/' + datap + '.dev', 'w') as wfile:
            wfile.write(cols)
            for idx in indices[int(0.8*len(lines)):int(0.9*len(lines))]:
                wfile.write(lines[idx])
        with open(datap + '/' + datap + '.test', 'w') as wfile:
            wfile.write(cols)
            for idx in indices[int(0.9*len(lines)):]:
                wfile.write(lines[idx])

for datap in data_list:
    split(datap)
