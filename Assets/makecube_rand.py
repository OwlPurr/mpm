import sys
import numpy as np
import random
c_row = 15

path_w = 'cube.txt'

with open(path_w, mode='w') as f:
    for i in range(c_row*c_row*c_row):
        x = 12*random.random() + 10
        y = 12*random.random() + 10
        z = 12*random.random() + 16
        print(f"頂点 {i+1}: x={x}, y={y}, z={z}")
        f.write(f"{(x-0.5)}\t{(y-0.5)}\t{(z-0.5)}\n")