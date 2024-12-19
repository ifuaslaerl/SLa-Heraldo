from random import seed
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler

from src.pipeline import Pipeline

def media(lista):
    """Calcula a média de uma lista de números"""
    return sum(lista) / len(lista)

def mediana(lista):
    """Calcula a mediana de uma lista de números"""
    sorted_lista = sorted(lista)
    n = len(sorted_lista)
    meio = n // 2
    return sorted_lista[meio]

def moda(lista):
    """Calcula a moda de uma lista de números"""
    counter = Counter(lista)
    max_freq = max(counter.values())
    return [item for item, freq in counter.items() if freq == max_freq][0]

if __name__ == "__main__":
    
    seed(144523)

    train = pd.read_csv("data/trabalho1/conjunto_de_treinamento.csv")
    test = pd.read_csv("data/trabalho1/conjunto_de_teste.csv")

    scalers = [StandardScaler(), MinMaxScaler(), MaxAbsScaler(), RobustScaler()]
    fill_categorical_strategys = [mediana]
    fill_numerical_strategys = [media, mediana]

    to_remove = ["inadimplente", "id_solicitante", "grau_instrucao"]
    selected_columns = [coluna for coluna in train.columns if not coluna in to_remove]

    plt.imshow(train[selected_columns].select_dtypes(exclude=['object']).corr())
    plt.show()

    for scaler in scalers:
        for index, fill_categorical_strategy in enumerate(fill_categorical_strategys):
            for jndex, fill_numerical_strategy in enumerate(fill_numerical_strategys):
                data_train, data_test, correlation_matrix = Pipeline(scaler=scaler,
                    fill_categorical_strategy=fill_categorical_strategy,
                    fill_numerical_strategy=fill_numerical_strategy,
                    select_collumns=selected_columns).process(train, test)

