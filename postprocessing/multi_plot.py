import os

import dill
import numpy as np
import pandas as pd
import plotly.graph_objs as go

rolling_factor = 5

#Kann man beliebig anpassen. Wenn z.b. nur der Median geplottet werden soll für eine sache gibt man da nur median an.
plots = {
    r"C:\workspace\Study_Project_Wind_Energy\Results\saves_random_repair": {
        "mean": {"name": "Mean Random Repair", "color": "rgb(250, 110, 110)"},
        "median": {"name": "Median Random Repair", "color": "rgb(199, 81, 110)"},
        "max": {"name": "Max Random Repair", "color": "rgb(143, 60, 100)"},
        "min": {"name": "Min Random Repair", "color": "rgb(88, 42, 80)"},
        "bars": {"name": "Min Random Repair", "color": "rgb(238, 41, 41)"}
    },
    r"C:\workspace\Study_Project_Wind_Energy\Results\saves_Simple_Repair": {
        "mean": {"name": "Mean Simple Repair", "color": "rgb(110, 113, 250)"},
        "median": {"name": "Median Simple Repair", "color": "rgb(0, 119, 209)"},
        "max": {"name": "Max Simple Repair", "color": "rgb(0, 106, 143)"},
        "min": {"name": "Min  Simple Repair", "color": "rgb(42, 85, 88)"},
        "bars": {"name": "50% Confidence Interval Simple Repair", "color": "rgb(238, 41, 41)"}
    },
    r"C:\workspace\Study_Project_Wind_Energy\Results\saves_smart_repair": {
        "mean": {"name": "Mean Smart Repair", "color": "rgb(110, 250, 113)"},
        "median": {"name": "Median Smart Repair", "color": "rgb(145, 181, 61)"},
        "max": {"name": "Max Random Smart", "color": "rgb(130, 119, 46)"},
        "min": {"name": "Min Random Smart", "color": "rgb(88, 68, 42)"},
        "bars": {"name": "50% Confidence Interval Smart", "color": "rgb(238, 41, 41)"}
    }}

#Ordner Angeben für den Gespeichert werden soll
save_path = r"C:\workspace\Study_Project_Wind_Energy\Results\compare_graphics\repairs"

def plot():
    plot_objects = []
    for key in plots:
        with open(os.path.join(key, "res.dill"), "rb") as file:
            res = dill.load(file)

        df = get_dataframe(res)
        df2 = df.copy().groupby("Generations")
        plot_attributes = plots[key]
        for attr in plot_attributes:
            if attr == "mean":
                mean = df2.mean()
                mean = pd.concat([mean, df2.quantile([0.25, 0.75]).unstack()], axis=1)
                mean["std"] = mean["Fitness"].rolling(rolling_factor).std()
                mean["avg"] = mean["Fitness"].rolling(rolling_factor).mean()
                plot_objects.append(go.Scatter(
                    name=plots[key][attr]["name"],
                    x=mean.index,
                    y=mean['Fitness'],
                    mode='lines',
                    line=dict(color=plots[key][attr]["color"]),
                ))
            elif attr == "median":
                median = df2.median()
                median["std"] = median["Fitness"].rolling(rolling_factor).std()
                median["avg"] = median["Fitness"].rolling(rolling_factor).mean()
                plot_objects.append(go.Scatter(
                    name=plots[key][attr]["name"],
                    x=median.index,
                    y=median['Fitness'],
                    mode='lines',
                    line=dict(color=plots[key][attr]["color"]),
                ))
            elif attr == "max":
                max = df2.max()
                max["std"] = max["Fitness"].rolling(rolling_factor).std()
                max["avg"] = max["Fitness"].rolling(rolling_factor).mean()
                plot_objects.append(go.Scatter(
                    name=plots[key][attr]["name"],
                    x=max.index,
                    y=max['Fitness'],
                    mode='lines',
                    line=dict(color=plots[key][attr]["color"]),
                ))
            elif attr == "min":
                min = df2.min()
                min["std"] = min["Fitness"].rolling(rolling_factor).std()
                min["avg"] = min["Fitness"].rolling(rolling_factor).mean()
                plot_objects.append(go.Scatter(
                    name=plots[key][attr]["name"],
                    x=min.index,
                    y=min['Fitness'],
                    mode='lines',
                    line=dict(color=plots[key][attr]["color"]),
                ))
            elif attr == "bars":
                bars = df2.mean()
                bars = pd.concat([bars, df2.quantile([0.25, 0.75]).unstack()], axis=1)
                bars["std"] = bars["Fitness"].rolling(rolling_factor).std()
                bars["avg"] = bars["Fitness"].rolling(rolling_factor).mean()
                plot_objects.append(go.Scatter(
                    name=plots[key][attr]["name"] + "Upper Bound",
                    x=bars.index,
                    y=bars[('Fitness', 0.75)],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False
                ))
                plot_objects.append(go.Scatter(
                    name=plots[key][attr]["name"] + "Lower Bound",
                    x=bars.index,
                    y=bars[('Fitness', 0.25)],
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    mode='lines',
                    fillcolor='rgba(68, 68, 68, 0.3)',
                    fill='tonexty',
                    showlegend=False
                ))

    fig = go.Figure(plot_objects)
    fig.update_layout(
        xaxis_title="Generationen",
        yaxis_title='Gewinne in Euro',
        title='Gewinne',
        hovermode="x"
    )
    fig.show()

    fig.write_html(os.path.join(save_path, "figure.html"))
    fig.write_image(os.path.join(save_path, "figure.svg"))


def get_dataframe(res):
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
    return df


if __name__ == '__main__':
    plot()
