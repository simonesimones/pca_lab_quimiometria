import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(

page_title="QuimioLab",

page_icon="🧪",

layout="wide"

)

st.title("🧪 QuimioLab UEPB")

st.markdown(
"### Plataforma Interativa para Ensino de Quimiometria"
)

perfil = st.sidebar.radio(

"Perfil",

["Graduação","Pós-graduação"]

)

st.sidebar.success(

f"Modo selecionado: {perfil}"

)

st.sidebar.markdown("---")

modulo = st.sidebar.selectbox(

"Escolha um módulo",

[
"Explorador de Dados",

"Pré-processamento",

"Reconhecimento de Padrões",

"Classificação",

"Calibração",

"Escape Room"

]

)

#####################################

if modulo=="Explorador de Dados":

    st.header("📊 Explorador de Dados")

    arquivo = st.file_uploader(

        "Carregar CSV ou XLSX",

        type=["csv","xlsx"]

    )

    if arquivo:

        if arquivo.name.endswith(".xlsx"):

            dados = pd.read_excel(arquivo)

        else:

            dados = pd.read_csv(arquivo)

        st.subheader("Dados")

        st.dataframe(dados)

        X = dados.select_dtypes(include="number")

        n_amostras = X.shape[0]

        n_variaveis = X.shape[1]

        st.subheader("Resumo")

        col1,col2 = st.columns(2)

        col1.metric(

            "Amostras",

            n_amostras

        )

        col2.metric(

            "Variáveis",

            n_variaveis

        )

        st.subheader(

            "Estatística Descritiva"

        )

        st.dataframe(

            X.describe()

        )

        st.subheader(

            "Histograma"

        )

        var = st.selectbox(

            "Escolha uma variável",

            X.columns

        )

        fig,ax = plt.subplots()

        ax.hist(

            X[var],

            bins=20

        )

        st.pyplot(fig)

        st.subheader(

            "Boxplot"

        )

        fig2,ax2 = plt.subplots()

        ax2.boxplot(

            X[var]

        )

        st.pyplot(fig2)

        st.subheader(

            "Correlação"

        )

        corr = X.corr()

        fig3,ax3 = plt.subplots(

            figsize=(7,7)

        )

        im = ax3.imshow(

            corr,

            aspect='auto'

        )

        plt.colorbar(im)

        st.pyplot(fig3)

        st.subheader(

            "Interpretação"

        )

        st.info(

f"""

Foram carregadas {n_amostras} amostras

e {n_variaveis} variáveis.

Analise a dispersão dos dados

e avalie a necessidade

de pré-processamento.

"""

)

        st.success(

"🏅 Badge desbloqueada: Explorador Químico"

)

#####################################

elif modulo=="Pré-processamento":

    st.header(

"🔬 Em desenvolvimento"

)

#####################################

elif modulo=="Reconhecimento de Padrões":

    st.header(

"📈 Em desenvolvimento"

)

#####################################

elif modulo=="Classificação":

    st.header(

"🎯 Em desenvolvimento"

)

#####################################

elif modulo=="Calibração":

    st.header(

"📐 Em desenvolvimento"

)

#####################################

elif modulo=="Escape Room":

    st.header(

"🔓 Em desenvolvimento"

)