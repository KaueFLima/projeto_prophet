import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error

def calcular_e_plotar(url, ano_corte):
    df = pd.read_csv(url)
    df['ds'] = pd.to_datetime(df['ds'])
    
    df_treino = df[df['ds'].dt.year <= ano_corte]
    df_teste = df[df['ds'].dt.year > ano_corte]
    
    modelo = Prophet(seasonality_mode='multiplicative')
    modelo.fit(df_treino)
    
    datas_futuras = modelo.make_future_dataframe(periods=len(df_teste), freq='MS')
    previsao = modelo.predict(datas_futuras)
    
    previsao_teste = previsao.iloc[len(df_treino):]
    mae = mean_absolute_error(df_teste['y'], previsao_teste['yhat'])
    
    plt.figure(figsize=(14, 7))
    plt.plot(df['ds'], df['y'], label='Dados Reais', color='black', marker='.', zorder=1)
    plt.plot(previsao['ds'], previsao['yhat'], label='Previsão', color='blue', linestyle='--', linewidth=2, zorder=2)
    plt.fill_between(previsao['ds'], previsao['yhat_lower'], previsao['yhat_upper'], color='blue', alpha=0.2)
    plt.axvline(x=pd.Timestamp(f'{ano_corte}-12-01'), color='red', linestyle='-', linewidth=2, label=f'Corte ({ano_corte})')
    
    plt.title(f'Previsão Prophet | Ano de Corte: {ano_corte} | MAE: {mae:,.2f}', fontsize=16, fontweight='bold')
    plt.xlabel('Ano', fontsize=12)
    plt.ylabel('Volume de Vendas', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

url = 'https://raw.githubusercontent.com/facebook/prophet/main/examples/example_retail_sales.csv'

calcular_e_plotar(url, 2004)
calcular_e_plotar(url, 2006)
calcular_e_plotar(url, 2008)