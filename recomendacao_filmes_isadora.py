

import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from surprise import Dataset, Reader, SVD, KNNBaseline, accuracy
from surprise.model_selection import KFold

sns.set_theme(style="whitegrid")
SEED = 42


URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
pasta = Path("ml-100k")

if not pasta.exists():
    urllib.request.urlretrieve(URL, "ml-100k.zip")
    with zipfile.ZipFile("ml-100k.zip") as z:
        z.extractall(".")

#user_id, item_id, rating, timestamp (separado por tabulação)
ratings = pd.read_csv(
    pasta / "u.data", sep="\t",
    names=["user_id", "item_id", "rating", "timestamp"],
)

#id, título e 19 colunas binárias de gênero (um filme pode ter vários gêneros)
GENEROS = [
    "Desconhecido", "Ação", "Aventura", "Animação", "Infantil", "Comédia",
    "Crime", "Documentário", "Drama", "Fantasia", "Film-Noir", "Terror",
    "Musical", "Mistério", "Romance", "Ficção científica", "Suspense",
    "Guerra", "Faroeste",
]
filmes = pd.read_csv(
    pasta / "u.item", sep="|", encoding="latin-1", header=None,
    usecols=[0, 1] + list(range(5, 24)), names=["item_id", "titulo"] + GENEROS,
)

print(f"Avaliações: {len(ratings):,}")
print(f"Usuários:   {ratings['user_id'].nunique():,}")
print(f"Filmes:     {ratings['item_id'].nunique():,}")

densidade = len(ratings) / (ratings["user_id"].nunique() * ratings["item_id"].nunique())
print(f"Densidade da matriz usuário x filme: {densidade:.2%}")

ratings.head()


distribuicao = (
    ratings["rating"].value_counts().sort_index()
    .rename_axis("Nota").reset_index(name="Quantidade")
)
distribuicao["Percentual (%)"] = (100 * distribuicao["Quantidade"] / len(ratings)).round(2)
display(distribuicao)

fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=distribuicao, x="Nota", y="Quantidade", color="green", ax=ax)
ax.set_title("Distribuição das notas")
ax.set_xlabel("Nota")
ax.set_ylabel("Quantidade de avaliações")
plt.tight_layout()
plt.show()

print(f"Nota média: {ratings['rating'].mean():.2f}")


por_usuario = ratings.groupby("user_id").size()
por_filme = ratings.groupby("item_id").size()

estatisticas = pd.DataFrame({
    "Avaliações por usuário": por_usuario.describe(),
    "Avaliações por filme": por_filme.describe(),
}).round(2)
display(estatisticas)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sns.histplot(por_usuario, bins=40, color="green", ax=axes[0])
axes[0].set_title("Avaliações por usuário")
axes[0].set_xlabel("Número de avaliações")
axes[0].set_ylabel("Quantidade de usuários")

sns.histplot(por_filme, bins=40, color="olive", ax=axes[1])
axes[1].set_title("Avaliações por filme")
axes[1].set_xlabel("Número de avaliações")
axes[1].set_ylabel("Quantidade de filmes")

plt.tight_layout()
plt.show()


mais_avaliados = (
    por_filme.rename("Avaliações").reset_index()
    .merge(filmes, on="item_id")
    .sort_values("Avaliações", ascending=False)
    .head(10)[["titulo", "Avaliações"]]
    .rename(columns={"titulo": "Título"})
    .reset_index(drop=True)
)
mais_avaliados



filmes["n_generos"] = filmes[GENEROS].sum(axis=1)
print(f"Média de gêneros por filme: {filmes['n_generos'].mean():.2f}")
display(
    filmes["n_generos"].value_counts().sort_index()
    .rename_axis("Nº de gêneros").reset_index(name="Filmes")
)

#uma linha por avaliação, com os gêneros do filme avaliado
ratings_gen = ratings.merge(filmes[["item_id"] + GENEROS], on="item_id")

tabela_generos = pd.DataFrame({
    "Filmes": filmes[GENEROS].sum(),
    "Avaliações": [(ratings_gen[g] == 1).sum() for g in GENEROS],
    "Nota média": [ratings_gen.loc[ratings_gen[g] == 1, "rating"].mean() for g in GENEROS],
})
tabela_generos["% dos filmes"] = (100 * tabela_generos["Filmes"] / len(filmes)).round(1)
tabela_generos["% das avaliações"] = (100 * tabela_generos["Avaliações"] / len(ratings)).round(1)
tabela_generos["Avaliações por filme"] = (tabela_generos["Avaliações"] / tabela_generos["Filmes"]).round(1)
tabela_generos["Nota média"] = tabela_generos["Nota média"].round(2)
tabela_generos = (
    tabela_generos.sort_values("Filmes", ascending=False)
    .rename_axis("Gênero").reset_index()
)
display(tabela_generos)

fig, axes = plt.subplots(1, 3, figsize=(16, 6))
paineis = [
    ("Filmes", "Quantidade de filmes por gênero", "steelblue"),
    ("Avaliações", "Quantidade de avaliações por gênero", "teal"),
    ("Nota média", "Nota média por gênero", "darkorange"),
]
for ax, (coluna, titulo, cor) in zip(axes, paineis):
    ordem = tabela_generos.sort_values(coluna, ascending=False)
    sns.barplot(data=ordem, y="Gênero", x=coluna, color=cor, ax=ax)
    ax.set_title(titulo)
    ax.set_ylabel("")
axes[2].set_xlim(1, 5)
plt.tight_layout()
plt.show()


reader = Reader(rating_scale=(1, 5))
dados = Dataset.load_from_df(ratings[["user_id", "item_id", "rating"]], reader)


def criar_modelos():
    return {
        "SVD": SVD(n_factors=50, n_epochs=25, random_state=SEED),
        "KNN por usuário": KNNBaseline(
            k=40,
            sim_options={"name": "pearson_baseline", "user_based": True},
            verbose=False,
        ),
    }



K = 10
LIMIAR_RELEVANTE = 4  #nota real >= 4 conta como "gostou"


def precisao_em_k(previsoes, k=K, limiar=LIMIAR_RELEVANTE):
    """Precisão média entre usuários com pelo menos k avaliações no teste."""
    por_usuario = defaultdict(list)
    for p in previsoes:
        por_usuario[p.uid].append((p.est, p.r_ui))

    precisoes = []
    for lista in por_usuario.values():
        if len(lista) < k:
            continue
        lista.sort(key=lambda par: par[0], reverse=True)
        acertos = sum(real >= limiar for _, real in lista[:k])
        precisoes.append(acertos / k)

    return np.mean(precisoes), len(precisoes)


kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
resultados = []

for fold, (trainset, testset) in enumerate(kf.split(dados), start=1):
    for nome, modelo in criar_modelos().items():
        modelo.fit(trainset)
        previsoes = modelo.test(testset)

        precisao, n_usuarios = precisao_em_k(previsoes)
        resultados.append({
            "Modelo": nome,
            "Fold": fold,
            "RMSE": accuracy.rmse(previsoes, verbose=False),
            "Precisão": precisao,
        })
    print(f"Fold {fold} concluído ({n_usuarios} usuários na Precisão)")

resultados = pd.DataFrame(resultados)
resultados.pivot(index="Fold", columns="Modelo", values=["RMSE", "Precisão"]).round(4)


resumo = (
    resultados.groupby("Modelo", sort=False)[["RMSE", "Precisão"]]
    .agg(["mean", "std"]).round(4)
)
resumo.columns = ["RMSE médio", "RMSE (desvio)", "Precisão média", "Precisão (desvio)"]
display(resumo)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

for ax, metrica in zip(axes, ["RMSE", "Precisão"]):
    sns.lineplot(data=resultados, x="Fold", y=metrica, hue="Modelo", marker="o", ax=ax)
    ax.set_title(f"{metrica} por rodada da validação cruzada")
    ax.set_xticks([1, 2, 3, 4, 5])

plt.tight_layout()
plt.show()


m = resumo["RMSE médio"], resumo["Precisão média"]
melhor_rmse = m[0].idxmin()
melhor_prec = m[1].idxmax()
vitorias_rmse = (
    resultados.pivot(index="Fold", columns="Modelo", values="RMSE").idxmin(axis=1)
    .value_counts().get(melhor_rmse, 0)
)
vitorias_prec = (
    resultados.pivot(index="Fold", columns="Modelo", values="Precisão").idxmax(axis=1)
    .value_counts().get(melhor_prec, 0)
)

print(f"Menor RMSE médio:      {melhor_rmse} "
      f"({m[0].min():.4f} contra {m[0].max():.4f}); melhor em {vitorias_rmse} de 5 rodadas.")
print(f"Maior Precisão:     {melhor_prec} "
      f"({m[1].max():.4f} contra {m[1].min():.4f}); melhor em {vitorias_prec} de 5 rodadas.")
