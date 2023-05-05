import os
import dill
import re
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

#This assumes the files are sorted by size

#df = pd.DataFrame(columns=["ID", "Spatial Resolution in Meters", "Fitness Value"])

nameMap = {"res_com_dynRuntime":"Dynamic Termination Criterion", "res_comp_fixRuntime":"100 Iterations"}


appr = []
results = []
resultsID = []
path = r"/Users/emily/Desktop/Workspace/Study_Project_Wind_Energy/results"
approaches = os.listdir(path)
for approach in approaches:
    elements_path = os.path.join(path, approach)
    elements = os.listdir(elements_path)
    for element in elements:
        res_path = os.path.join(elements_path, element, "res.dill")
        IDs = list(filter(str.isdigit, element))
        ID = ""
        for i in IDs:
            ID += i
        ID = int(ID)
        resultsID.append(ID)
        appr.append(approach)
        with open(res_path, "rb") as file:
            res = dill.load(file)
            results.append(res)

# for element in approaches:
#     res_path = os.path.join(path, element, "res.dill")
#     IDs = list(filter(str.isdigit, element))
#     ID = ""
#     for i in IDs:
#         ID += i
#     ID = int(ID)
#     resultsID.append(ID)
#     with open(res_path, "rb") as file:
#         res = dill.load(file)
#         results.append(res)
resultsF = [-res.F[0] for res in results]

appr_rep = []

for i in appr:
    appr_rep.append(nameMap[i])

df = pd.DataFrame(dict(x=resultsID, y=resultsF, appr=appr_rep))
df.sort_values(by=['x'], inplace=True)
df.rename({"x": "Spatial Resolution in Meters", "y": "Fitness Value", "appr":"Approach"}, axis=1, inplace=True)
#fig2 = px.line(df, x="x", y="y", title="Sorted Input")
#fig2.show()
#fig = px.scatter(x=resultsID, y=resultsF, trendline="ols")
#fig.show()

fig = px.line(df, x="Spatial Resolution in Meters", y="Fitness Value", color="Approach", text="Spatial Resolution in Meters", title="Fitness Value over Spatial Resolution")
fig.update_traces(textposition="bottom right")
fig.show()
