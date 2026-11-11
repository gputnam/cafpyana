import uproot
import matplotlib.pyplot as plt
import io
import json

# 1. Open the ROOT file
file = uproot.open("/home/nathanielerowe/Downloads/MINOS_data.root")

# 2. Access the TGraph object
# Replace 'my_graph' with the name of your graph in the ROOT file
graph = file["minosPlus90CL_0"]

# 3. Access the underlying arrays
# ROOT's TGraph stores coordinates in fX and fY members
x_values = graph.member("fX")
y_values = graph.member("fY")

# Note: TGraph might store more points in the array than are actually used (fNpoints)
# It's safest to slice the arrays to the actual point count
n_points = graph.member("fNpoints")
x_values = x_values[:n_points]
y_values = y_values[:n_points]

data = {'x': list(x_values), 
        'y': list(y_values)}

#data = {'x': [str(x) for x in x_values], 
#        'y': [str(y) for y in y_values]}


with io.open('minos.json', 'w', encoding='utf8') as outfile:
    out_str = json.dumps(data,
                      indent=4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(out_str)
