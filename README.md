# Sistemas de Recomendação: filmes MovieLens 100K

Comparação de dois modelos de filtragem colaborativa, **SVD** (fatorização matricial) e **KNN baseado em usuários**, aplicados ao conjunto de dados [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/), que reúne 100.000 avaliações (notas de 1 a 5) feitas por 943 usuários sobre 1.682 filmes.

O trabalho foi desenvolvido em um notebook do Google Colab ([abrir original](https://colab.research.google.com/drive/1IHbVhnBl9NZQQe1d75PqsIKlgTzV2Ysg)); o arquivo `.py` é a exportação desse notebook.

## Etapas do projeto

1. **Preparação dos dados**: download automático do MovieLens 100K, leitura das avaliações (`u.data`) e dos filmes com seus 19 gêneros (`u.item`), e cálculo da densidade da matriz usuário × filme.
2. **Análise exploratória**
   - distribuição das notas;
   - número de avaliações por usuário e por filme (distribuições com cauda longa) e os filmes mais avaliados;
   - gêneros: quantidade de filmes, avaliações, avaliações por filme e nota média de cada gênero (um filme pode ter mais de um gênero).
3. **Treinamento dos modelos**, usando apenas as notas
   - **SVD**: 50 fatores latentes, 25 épocas.
   - **KNN por usuário** (`KNNBaseline`): 40 vizinhos, similaridade `pearson_baseline`.
4. **Avaliação**: validação cruzada com 5 folds (80% treino, 20% teste), com os dois modelos treinados e testados nas mesmas divisões.
   - **RMSE**: diferença entre notas reais e previstas (menor é melhor).
   - **Precisão@10**: proporção de filmes entre os 10 melhores ranqueados com nota real ≥ 4. Considera apenas usuários com pelo menos 10 avaliações no teste e mede a qualidade do ranking só entre filmes que o usuário avaliou, não no catálogo inteiro (maior é melhor).
5. **Comparação e discussão** dos resultados por rodada e na média.

## Autora

Isadora Fumagali — [@isadorafumagali](https://github.com/isadorafumagali)
