import streamlit as st
import pandas as pd
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt

# Carregar o modelo treinado
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_sarima_model.pkl')
        return model
    except Exception as e:
        st.error(f"Erro ao carregar o modelo: {e}")
        return None

best_model = load_model()

# Função para fazer previsões com o modelo
def make_prediction(model, start, periods):
    try:
        pred = model.predict(start=start, end=start + periods - 1, dynamic=False)
        return pred
    except Exception as e:
        st.error(f"Erro ao fazer a previsão: {e}")
        return None

# Criar uma aplicação Streamlit
st.title('Previsão de Acidentes')
st.write('Esta é uma aplicação para prever o número de acidentes por mês.')

# Carregar os dados
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('df_nordeste.csv')
        df['data_inversa'] = pd.to_datetime(df['data_inversa'], format='%Y-%m-%d', errors='coerce')
        df['ano'] = df['data_inversa'].dt.year
        df['mes'] = df['data_inversa'].dt.month
        acidentes_por_mes_ano = df.groupby(['ano', 'mes']).size().reset_index(name='contagem')
        acidentes_por_mes_ano['data'] = pd.to_datetime(pd.PeriodIndex(year=acidentes_por_mes_ano['ano'], month=acidentes_por_mes_ano['mes'], freq='M').to_timestamp())
        acidentes_por_mes_ano = acidentes_por_mes_ano.sort_values(by='data').set_index('data')
        return acidentes_por_mes_ano
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {e}")
        return pd.DataFrame()

data_load_state = st.text('Carregando dados...')
acidentes_por_mes_ano = load_data()
data_load_state.text('Dados carregados com sucesso!')

if not acidentes_por_mes_ano.empty:
    # Seleção do número de meses para prever no futuro
    num_months = st.slider('Selecione o número de meses para prever:', min_value=1, max_value=24, value=6)

    # Botão para fazer previsão
    if st.button('Fazer Previsão'):
        if best_model is not None:
            # Último ponto de dados disponível
            last_data_point = acidentes_por_mes_ano.index.max()
            
            # Calcular a data de início da previsão
            start_prediction = last_data_point + pd.DateOffset(months=1)

            # Fazer previsão
            prediction = make_prediction(best_model, start=len(acidentes_por_mes_ano), periods=num_months)
            if prediction is not None:
                # Criar índice de datas para a previsão
                prediction_index = pd.date_range(start=start_prediction, periods=num_months, freq='M')

                # Ajustar o índice da previsão
                prediction.index = prediction_index

                # Criar dataframe com as datas e previsões
                prediction_df = pd.DataFrame({'data': prediction_index, 'previsao': prediction.values})

                # Arredondar a previsão para números inteiros
                prediction_df['previsao'] = prediction_df['previsao'].round().astype(int)

                # Exibir previsão
                st.write("Previsões ajustadas:")
                st.write(prediction_df)

                # Visualizar apenas as previsões
                plt.figure(figsize=(10, 6))
                plt.plot(prediction_df['data'], prediction_df['previsao'], label='Previsão de Acidentes', color='green', marker='o')
                plt.xlabel('Data')
                plt.ylabel('Número de Acidentes')
                plt.title('Previsão de Acidentes')
                plt.grid(True)
                plt.legend()
                st.pyplot(plt)
            else:
                st.error("Erro ao fazer a previsão.")
        else:
            st.error("O modelo não está disponível para fazer previsões.")
