import pandas as pd
from sklearn.preprocessing import RobustScaler

def tratar_dados_nao_numericos(df: pd.DataFrame, colunas_categoricas: list, colunas_finais=None) -> pd.DataFrame:
    """ Transforma os dados categóricos em uma estrutura numérica consistente. """
    df_dummies = pd.get_dummies(df[colunas_categoricas], drop_first=True)

    # Garantir que as colunas estejam alinhadas com o conjunto de treino
    if colunas_finais is not None:
        for col in colunas_finais:
            if col not in df_dummies:
                df_dummies[col] = 0
        df_dummies = df_dummies[colunas_finais]
    
    # Remover colunas categóricas originais e adicionar dummies
    df = pd.concat([df.drop(columns=colunas_categoricas), df_dummies], axis=1)
    return df

def normalizar_dados(df: pd.DataFrame, colunas_para_normalizar: list, scaler=None):
    """ Normaliza os dados com RobustScaler. """
    if scaler:
        df_normalizado = scaler.fit_transform(df[colunas_para_normalizar])
        df_normalizado = pd.DataFrame(df_normalizado, columns=colunas_para_normalizar, index=df.index)
        return pd.concat([df.drop(columns=colunas_para_normalizar), df_normalizado], axis=1), scaler
    
    return df, scaler

def processar_dados(path, train=True, colunas_finais=None, scaler=None):
    """ Prepara os dados para treino ou teste. """
    # Carregar os dados
    data = pd.read_csv(path)

    # Tratar dados categóricos
    colunas_categoricas = data.select_dtypes(exclude=[int, float, bool]).columns.tolist()
    data = tratar_dados_nao_numericos(data, colunas_categoricas, colunas_finais)
    
    # Preencher NaN
    if "meses_na_residencia" in data.columns:
        data["meses_na_residencia"] = data["meses_na_residencia"].fillna(data["meses_na_residencia"].mean())
    data = data.fillna(0)

    # Normalizar os dados
    if train:
        colunas_excluidas = ["inadimplente", "id_solicitante"]
        colunas_para_normalizar = [col for col in data.columns if col not in colunas_excluidas]
        data, scaler = normalizar_dados(data, colunas_para_normalizar)
        colunas_finais = data.columns  # Salvar colunas finais
    else:
        colunas_excluidas = ["id_solicitante"]
        colunas_para_normalizar = [col for col in data.columns if col not in colunas_excluidas]
        data, _ = normalizar_dados(data, colunas_para_normalizar, scaler)

    return data, colunas_finais, scaler

def main():
    """ Execução principal do código. """
    # Processar os dados de treino
    data_train, colunas_finais, scaler = processar_dados("data/trabalho1/conjunto_de_treinamento.csv", train=True)

    # Processar os dados de teste com as mesmas colunas e escalador
    data_test, _, _ = processar_dados("data/trabalho1/conjunto_de_teste.csv", train=False, colunas_finais=colunas_finais, scaler=scaler)

    # Salvar os dados tratados (opcional)
    data_train.to_csv("data/trabalho1/dados_treinamento_tratados.csv", index=False)
    data_test.to_csv("data/trabalho1/dados_teste_tratados.csv", index=False)

    return data_train, data_test

if __name__ == "__main__":
    data_train, data_test = main()
