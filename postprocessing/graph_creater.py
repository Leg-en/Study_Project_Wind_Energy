import os

import dill
import numpy as np
import pandas as pd
import plotly.graph_objs as go



rolling_factor = 5
res_path = r"C:\uni_workspace\Study_Project_Wind_Energy\Project_Arrays\saves_smart_repair"
with open(os.path.join(res_path, "res.dill"), "rb") as file:
    res = dill.load(file)




fitness_vals = []
for iteration in res.history:
    x = []
    for item in iteration.pop:
        x.append(item.F[0])
    fitness_vals.append(x)
np_fitness = np.asarray(fitness_vals)

# Richtige Generation
d = {}
d["Generations"] = []
d["Fitness"] = []
for i in range(np_fitness.shape[0]):
    for j in range(np_fitness.shape[1]):
        d["Generations"].append(i)
        d["Fitness"].append(np_fitness[i, j])

df = pd.DataFrame.from_dict(d)
df["Fitness"] = df["Fitness"] * -1
df2 = df.copy(deep=True)

df["std"] = df["Fitness"].rolling(rolling_factor).std()
df["avg"] = df["Fitness"].rolling(rolling_factor).mean()

df2_ = df2.groupby("Generations")

min = df2_.min()
min["std"] = min["Fitness"].rolling(rolling_factor).std()
min["avg"] = min["Fitness"].rolling(rolling_factor).mean()

max = df2_.max()
max["std"] = max["Fitness"].rolling(rolling_factor).std()
max["avg"] = max["Fitness"].rolling(rolling_factor).mean()

mean = df2_.mean()
# mean2 = mean.copy(deep=True)
mean = pd.concat([mean, df2_.quantile([0.25, 0.75]).unstack()], axis=1)
mean["std"] = mean["Fitness"].rolling(rolling_factor).std()
mean["avg"] = mean["Fitness"].rolling(rolling_factor).mean()

median = df2_.median()
median["std"] = median["Fitness"].rolling(rolling_factor).std()
median["avg"] = median["Fitness"].rolling(rolling_factor).mean()

# f = pd.concat([mean2, df2_.quantile([0.25,0.75]).unstack()], axis=1)

fig = go.Figure([
    #    go.Scatter(
    #        name='Gewinn',
    #        x=df["Generations"],
    #        y=df['Fitness'],
    #        mode='lines',
    #        line=dict(color='rgb(31, 119, 180)'),
    #    ),
    go.Scatter(
        name='Gewinn Min',
        x=min.index,
        y=min['Fitness'],
        mode='lines',
        line=dict(color='rgb(31, 119, 180)'),
    ),
    go.Scatter(
        name='Gewinn Max',
        x=max.index,
        y=max['Fitness'],
        mode='lines',
        line=dict(color='rgb(238, 41, 41)'),
    ),
    go.Scatter(
        name='Gewinn Mean',
        x=mean.index,
        y=mean['Fitness'],
        mode='lines',
        line=dict(color='rgb(128, 255, 0)'),
    ),
    go.Scatter(
        name='Gewinn Median',
        x=median.index,
        y=median['Fitness'],
        mode='lines',
        line=dict(color='rgb(255, 153, 255)'),
    ),

    go.Scatter(
        name='Upper Bound',
        x=mean.index,
        y=mean[('Fitness', 0.75)],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=0),
        showlegend=False
    ),
    go.Scatter(
        name='Lower Bound',
        x=mean.index,
        y=mean[('Fitness', 0.25)],
        marker=dict(color="#444"),
        line=dict(width=0),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )

])
fig.update_layout(
    xaxis_title="Generationen",
    yaxis_title='Gewinne in Euro',
    title='Gewinne',
    hovermode="x"
)
fig.show()
save_path = os.path.join(res_path, "images_csv")
os.mkdir(save_path)
df.to_csv(os.path.join(save_path, "dataframe.csv"))
mean.to_csv(os.path.join(save_path, "mean.csv"))
min.to_csv(os.path.join(save_path, "min.csv"))
max.to_csv(os.path.join(save_path, "max.csv"))
fig.write_html(os.path.join(save_path, "figure.html"))
fig.write_image(os.path.join(save_path, "figure.svg"))
