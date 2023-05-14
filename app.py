import streamlit as st

import numpy as np
import pandas as pd

from cm_func import *

st.set_page_config(
    page_title='Когнітивне моделювання',
    page_icon='🎓',
    layout='wide'
)

# to remove +/- from number input widgets
st.markdown("""
    <style>
        button.step-up {display: none;}
        button.step-down {display: none;}
        div[data-baseweb] {border-radius: 4px;}
    </style>""",
    unsafe_allow_html=True
)

st.write("# Когнітивне моделювання")

with st.sidebar.header('1. Виберіть .xlsx файл'):
    uploaded_file = st.sidebar.file_uploader("Виберіть .xlsx файл з когнітивною картою", type=["xlsx"])
    st.sidebar.markdown("""
        [Приклад необхідного файлу](https://github.com/ZaikaBohdan/datasetsforlabs/blob/main/sa_lab6_raw_input.xlsx?raw=true)
    """)

if uploaded_file is not None:
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Когнітивна карта <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    st.write("## Когнітивна карта")
    
    c1, c2 = st.columns(2) 

    cogn_map = pd.read_excel(uploaded_file, index_col=0)

    c1.write("### Матриця")
    c1.dataframe(cogn_map)

    c1.write("### Граф")
    graph = build_graph(cogn_map, c1)

    c2.write("### Стійкість")

    stab_vals = [
        check_perturbation_stability(cogn_map), 
        check_numerical_stability(cogn_map), 
        check_structural_stability(cogn_map, graph)
        ]
    stab_df = pd.DataFrame(
        [yes_no(val) for val in stab_vals], 
        index = ['За збуренням', 'Чисельна', 'Структурна'],
        columns=['Так/Ні']
        )

    c2.dataframe(stab_df.T)

    eigvals_list = [str(val).strip('()').replace('j', 'i') for val in eigvals(cogn_map)]
    
    c2.write(f"### Власні числа (max|λ| = {get_spectral_radius(cogn_map): .2f})")
    c2.dataframe(pd.Series(eigvals_list, name='Власні числа'), height=205)

    c2.write(f"### Парні цикли")
    if stab_vals[2]:
        c2.write('Відсутні')
    else:
        c2.dataframe(find_even_cycles(cogn_map, graph)[0], height=205)

    
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Імпульсивне моделювання <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    with st.sidebar.header('2. Параметри імпульсного моделювання'):
        st.sidebar.markdown("#### 2.1. Початковий стан")
        V = np.zeros(cogn_map.shape[0])
        v_cols = st.sidebar.columns(2)
        st.sidebar.markdown("#### 2.2. Початковий імпульс")
        P = np.zeros(cogn_map.shape[0])
        p_cols = st.sidebar.columns(2)
        t = st.sidebar.number_input('Кількість ітерацій', min_value=1, value=5)
        for i in range(cogn_map.shape[0]):
            V[i] = v_cols[i%2].number_input(f'v{i+1}', min_value=-1.0, max_value=1.0, value=0.0)
            P[i] = p_cols[i%2].number_input(f'p{i+1}', min_value=-1.0, max_value=1.0, value=0.0)
        imp_mod_button = st.sidebar.button('Виконати')

    if imp_mod_button:
        st.write("## Імпульсне моделювання")
        impulse_model(t, V, P, cogn_map)

else:
    st.info('Виберіть .xlsx файл з вхідними даними у боковому вікні зліва.')