""" Módulo feito para fit e test dos dados. """

# Bibliotecas usadas
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import treat_data

# Testar Varios parametros para o SGDClassifier

def sgd():
    """ Testa todos SGDClassifier que quero testar. """
    sgd_options = []
    sgd_loss = ["log_loss", "perceptron", "squared_error"]
    sgd_shuffle = [True, False]
    sgd_learning_rate = ["constant", "adaptive"]
    sgd_etao = [1, 10, 100, 1000]

    for loss in sgd_loss:
        for shuffle in sgd_shuffle:
            for learning_rate in sgd_learning_rate:
                for etao in sgd_etao:
                    sgd_options.append(SGDClassifier(
                                            eta0=etao,
                                            loss=loss,
                                            shuffle=shuffle,
                                            learning_rate=learning_rate))

    print(f"{len(sgd_options)} opções de SGD. ")
    return sgd_options

def knn(N=2000):
    """ Testa todos KNNClassifier que quero testar. """

    knn_options = []
    knn_n_neighbours = []
    n = 1
    while(n < N):
        knn_n_neighbours.append(n)
        n*=3

    knn_wheighs = ["uniform", "distance"]
    metrics = ["euclidean", "manhattan"]

    for n_neighbours in knn_n_neighbours:
        for wheigh in knn_wheighs:
            for metric in metrics:
                knn_options.append(KNeighborsClassifier(
                    n_neighbors=n_neighbours,
                    weights=wheigh,
                    metric=metric
                ))

    print(f"{len(knn_options)} opções para knn.")
    return knn_options

# Metricas para o Knn

def main():
    """ Execução principal do código. """

    # Selecionar os possíveis algorítimos de aprendizado de máquina que podemos usar

    modelos = []

    modelos += sgd()
    modelos += knn()
    modelos += [LogisticRegressionCV()]

    print(f"Testando {len(modelos)} modelos de aprendizado de máquina. ")

    # Receber dados

    data_train = pd.read_csv("data/trabalho1/dados_treinamento_tratados.csv")
    data_test = pd.read_csv("data/trabalho1/dados_teste_tratados.csv")

    y = data_train["inadimplente"]
    colunas = [coluna for coluna in data_train.columns if coluna != "inadimplente"]
    x = data_train[colunas] 

    # Separar parte de treino real, teste e validação dentro do conjunto de treino

    x_treino, x_validacao, y_treino, y_validacao = train_test_split(x, y, test_size=0.1, random_state=1322)

    x_treino, x_teste, y_treino, y_teste = train_test_split(x_treino, y_treino, test_size=0.1, random_state=1322)

    # Dividido em proporção 0.81, 0.09, 0.10 em treino, teste e validação

    resultados = []
    for modelo in modelos:
        modelo.fit(x_treino, y_treino)
        pred = modelo.predict(x_teste)
        precisao = accuracy_score(pred, y_teste)
        resultados.append((precisao, modelo))

    resultados.sort(key= lambda x: x[0], reverse=True)

    resultados = resultados[:9]
    result_final = []
    for porcentagem , modelo in resultados:
        pred = modelo.predict(x_validacao)
        precisao = accuracy_score(pred, y_validacao)
        result_final.append((precisao, porcentagem, modelo))

    result_final.sort(key= lambda x: x[0], reverse=True)

    for pres, percent, model in result_final:
        print(pres, percent, model)
        pred = modelo.predict(data_test)
        precisao = accuracy_score(pred, len(pred)*[1])

if __name__ == "__main__":
    main()
