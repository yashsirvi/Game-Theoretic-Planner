import numpy as np
import csv
# import pandas as pd
dt = 0.5
n_steps = 10
# track_waypoints = np.array([[0,0], [2.5, 5], [4, -2], [6, 9], [1, 4],[0, 0]])
with open("Spielberg_centerline.csv") as f:
    data = list(csv.reader(f))
    li = []
    for i in data[1:]:
        li.append([float(i[0]), float(i[1])])
    track_waypoints = np.array(li)
        
track_width = 0.3

collision_radius = 0.1
v_max = 5
a_max = 20

