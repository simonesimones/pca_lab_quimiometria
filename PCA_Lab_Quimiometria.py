import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from scipy.signal import savgol_filter
from scipy.cluster.hierarchy import linkage, dendrogram


st.set_page_config(
    page_title="PCA Lab Quimiometria",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 PCA Lab — Reconhecimento de Padrões")
st.caption("Interface didática para PCA, HCA e interpretação de agrupamentos em quimiometria.")

with st.expander("📘 Como usar em aula", expanded=False):
    st.markdown(
        """
        1. Carregue uma matriz em **Excel (.xlsx)** ou **CSV**.  
        2. A primeira coluna pode conter o nome das amostras.  
        3. Se houver uma coluna de classe/grupo, você pode usá-la apenas para colorir o gráfico.  
        4. Escolha o pré-processamento.  
        5. Clique em **Executar PCA**.  
        6. Interprete scores, loadings, variância explicada e HCA.  

        Para o Escape Room, use o **Modo desafio** para esconder as classes.
        """
    )

st.sidebar.header("⚙️ Configurações")
modo_desafio = st.sidebar.checkbox("Modo desafio: esconder classes", value=False)
mostrar_hca = st.sidebar.checkbox("Mostrar HCA / dendrograma", value=True)
mostrar_loadings = st.sidebar.checkbox("Mostrar gráfico de loadings", value=True)

arquivo = st.file_uploader("📁 Carregue sua matriz espectral (.xlsx ou .csv)", type=["xlsx", "csv"])


def read_file(file):
    if file.name.lower().endswith(".xlsx"):
        return pd.read_excel(file)
    return pd.read_csv(file)


def snv(X):
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1
    return (X - mean) / std


def preprocess(X, method):
    X = np.asarray(X, dtype=float)
    if method == "Nenhum":
        return X
    if method == "Centralização na média":
        return X - np.mean(X, axis=0)
    if method == "Autoescalamento (média 0, variância 1)":
        return StandardScaler().fit_transform(X)
    if method == "Normalização por amostra (área/vetor)":
        return normalize(X, norm="l2")
    if method == "Min-Max (0 a 1)":
        return MinMaxScaler().fit_transform(X)
    if method == "SNV":
        return snv(X)
    if method == "Derivada Savitzky-Golay 1ª derivada":
        window = 5 if X.shape[1] >= 5 else X.shape[1] - (1 - X.shape[1] % 2)
        if window < 3:
            st.warning("Poucas variáveis para Savitzky-Golay. Usando centralização.")
            return X - np.mean(X, axis=0)
        if window % 2 == 0:
            window -= 1
        return savgol_filter(X, window_length=window, polyorder=2, deriv=1, axis=1)
    return X


if arquivo is None:
    st.info("Carregue um arquivo para iniciar. Você pode usar a matriz do Escape Room ou qualquer matriz numérica.")
    st.stop()

try:
    dados = read_file(arquivo)
except Exception as e:
    st.error(f"Não consegui ler o arquivo: {e}")
    st.stop()

st.subheader("1) Dados carregados")
st.dataframe(dados, use_container_width=True)

colunas = list(dados.columns)

col1, col2, col3 = st.columns(3)
with col1:
    coluna_amostra = st.selectbox("Coluna com nomes das amostras", options=["Usar índice"] + colunas, index=0)
with col2:
    coluna_classe = st.selectbox("Coluna com classe/grupo (opcional)", options=["Nenhuma"] + colunas, index=0)
with col3:
    preprocessamento = st.selectbox(
        "Pré-processamento",
        options=[
            "Autoescalamento (média 0, variância 1)",
            "Centralização na média",
            "Nenhum",
            "Normalização por amostra (área/vetor)",
            "Min-Max (0 a 1)",
            "SNV",
            "Derivada Savitzky-Golay 1ª derivada",
        ],
        index=0
    )

excluir = []
if coluna_amostra != "Usar índice":
    excluir.append(coluna_amostra)
if coluna_classe != "Nenhuma":
    excluir.append(coluna_classe)

variaveis_numericas = [c for c in dados.select_dtypes(include=[np.number]).columns if c not in excluir]
variaveis = st.multiselect("Variáveis usadas na PCA", options=variaveis_numericas, default=variaveis_numericas)

if len(variaveis) < 2:
    st.error("Selecione pelo menos duas variáveis numéricas para executar a PCA.")
    st.stop()

if coluna_amostra == "Usar índice":
    nomes = dados.index.astype(str).tolist()
else:
    nomes = dados[coluna_amostra].astype(str).tolist()

classes = None
if coluna_classe != "Nenhuma" and not modo_desafio:
    classes = dados[coluna_classe].astype(str).tolist()

if st.button("▶️ Executar PCA", type="primary"):
    X = dados[variaveis].copy()
    X = X.apply(pd.to_numeric, errors="coerce")

    if X.isna().any().any():
        st.warning("Há valores ausentes ou não numéricos. Eles foram substituídos pela média da respectiva variável.")
        X = X.fillna(X.mean())

    Xp = preprocess(X.values, preprocessamento)

    n_comp = min(5, Xp.shape[0], Xp.shape[1])
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(Xp)
    loadings = pca.components_.T

    st.success("PCA executada com sucesso.")

    st.subheader("2) Variância explicada")
    var_df = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(n_comp)],
        "Variância explicada (%)": np.round(pca.explained_variance_ratio_ * 100, 2),
        "Variância acumulada (%)": np.round(np.cumsum(pca.explained_variance_ratio_) * 100, 2)
    })
    st.dataframe(var_df, use_container_width=True)

    fig_var, ax_var = plt.subplots(figsize=(7, 4))
    ax_var.bar(var_df["Componente"], var_df["Variância explicada (%)"])
    ax_var.set_ylabel("Variância explicada (%)")
    ax_var.set_title("Variância explicada por componente")
    st.pyplot(fig_var)

    st.subheader("3) Gráfico de Scores — mapa das amostras")
    scores_df = pd.DataFrame(scores[:, :2], columns=["PC1", "PC2"])
    scores_df.insert(0, "Amostra", nomes)
    if coluna_classe != "Nenhuma":
        scores_df.insert(1, "Classe", dados[coluna_classe].astype(str).tolist())
    st.dataframe(scores_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    if classes is None:
        ax.scatter(scores[:, 0], scores[:, 1])
    else:
        unique_classes = sorted(set(classes))
        for cl in unique_classes:
            idx = [i for i, c in enumerate(classes) if c == cl]
            ax.scatter(scores[idx, 0], scores[idx, 1], label=cl)
        ax.legend(title="Classe")

    for i, nome in enumerate(nomes):
        ax.text(scores[i, 0], scores[i, 1], nome, fontsize=9)

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA — Scores")
    st.pyplot(fig)

    if mostrar_loadings:
        st.subheader("4) Loadings — variáveis responsáveis pela separação")
        load_df = pd.DataFrame(loadings[:, :2], index=variaveis, columns=["PC1", "PC2"])
        st.dataframe(load_df, use_container_width=True)

        fig_l, ax_l = plt.subplots(figsize=(8, 6))
        ax_l.scatter(load_df["PC1"], load_df["PC2"])
        for var in load_df.index:
            ax_l.text(load_df.loc[var, "PC1"], load_df.loc[var, "PC2"], str(var), fontsize=9)
        ax_l.axhline(0, linewidth=0.8)
        ax_l.axvline(0, linewidth=0.8)
        ax_l.set_xlabel("Loading PC1")
        ax_l.set_ylabel("Loading PC2")
        ax_l.set_title("PCA — Loadings")
        st.pyplot(fig_l)

    if mostrar_hca:
        st.subheader("5) HCA — agrupamento hierárquico")
        metodo = st.selectbox("Método de ligação do HCA", ["ward", "complete", "average", "single"], index=0)
        Z = linkage(Xp, method=metodo)
        fig_h, ax_h = plt.subplots(figsize=(10, 5))
        dendrogram(Z, labels=nomes, ax=ax_h)
        ax_h.set_title(f"Dendrograma — método {metodo}")
        ax_h.set_ylabel("Distância")
        st.pyplot(fig_h)

    st.subheader("💡 Perguntas para interpretação")
    st.markdown(
        """
        - Quantos grupos aparecem no gráfico de scores?  
        - Existe alguma amostra isolada ou suspeita?  
        - O HCA confirma os agrupamentos observados na PCA?  
        - Quais variáveis têm maior contribuição nos loadings?  
        - A amostra desconhecida pertence a qual grupo?
        """
    )

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        scores_df.to_excel(writer, index=False, sheet_name="Scores")
        var_df.to_excel(writer, index=False, sheet_name="Variancia")
        if mostrar_loadings:
            load_df.to_excel(writer, sheet_name="Loadings")
    st.download_button(
        "⬇️ Baixar resultados em Excel",
        data=buffer.getvalue(),
        file_name="resultados_pca.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
