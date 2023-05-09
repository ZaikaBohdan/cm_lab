import numpy as np
import pandas as pd

import networkx as nx
import matplotlib
import matplotlib.pyplot as plt

import streamlit as st
import altair as alt


yes_no = lambda x: 'Так' if x else 'Ні'

def build_graph(df, st_place):
    G = nx.DiGraph()

    for el in df.index:
        G.add_node(el)

    color = lambda w: 'b' if w > 0 else 'o'
    edges = [(ind, col, {"weight" :df.loc[ind, col], 'color': color(df.loc[ind, col])}) for ind in df.index for col in df.columns if df.loc[ind, col]!=0]
    weights = [row[2]['weight']*15 for row in edges]
    G.add_edges_from(edges)

    pos = nx.circular_layout(G)
    fig = plt.figure(figsize=(40, 15))
    plt.gca().set_facecolor('lightgray')
    nx.draw_networkx(
        G,
        pos, 
        # nodes
        node_size=5000,
        font_color='white',
        font_size=30,
        # edges
        connectionstyle='arc3, rad = 0.1',
        width=5,
        edge_color=weights,
        edge_cmap=plt.cm.RdYlBu,
        arrowsize=30
        )
    plt.colorbar(
        plt.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(vmin=-1, vmax=1), 
            cmap=plt.cm.RdYlBu
            ),
        pad=0.01
        )
    st_place.pyplot(fig)

    return G

def eigvals(df):
    return np.linalg.eigvals(np.array(df))

def get_spectral_radius(df):
    return np.max(np.absolute(eigvals(df)))

def check_perturbation_stability(df):
    return get_spectral_radius(df) <= 1

def check_numerical_stability(df):
    return get_spectral_radius(df) < 1

def find_even_cycles(df, G):
    cycles = pd.Series([c for c in nx.simple_cycles(G)], name='Парні цикли')
    cycles = cycles.apply(lambda row: row+[row[0]])

    is_even = lambda nodes: sum([df.loc[edge[0], edge[1]] < 0 for edge in zip(nodes[:-1], nodes[1:])]) // 2 != 1
    even_mask = cycles.apply(is_even)
    
    return cycles[even_mask].reset_index(drop=True), cycles

def check_structural_stability(df, G):
    return find_even_cycles(df, G)[0].shape[0] == 0

def impulse_model(t, Q, df):
    title = f'Q = {list(Q)}'

    A = np.array(df)
    x_i = [np.zeros(A.shape[0]), np.zeros(A.shape[0])]
    for _ in range(t):
        x_new = x_i[-1] + np.matmul(A, x_i[-1] - x_i[-2]) + Q
        x_i.append(x_new)
        Q = np.zeros(A.shape[0])
    
    source = pd.DataFrame(
        x_i,
        columns=[f'e{i}' for i in range(A.shape[0])]
        )
    source = source.reset_index().melt('index', var_name='category', value_name='y')

    line_chart = alt.Chart(source).mark_line().encode(
        alt.X('index', title='t'),
        alt.Y('y', title='x(t)'),
        color='category:N'
    ).properties(title=title)

    st.altair_chart(line_chart, use_container_width=True)