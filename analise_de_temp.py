import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

def carregar_datasets():
    df_treino = pd.read_csv('dados_temperatura_treino.csv')
    df_validacao = pd.read_csv('dados_temperatura_validacao.csv')
    
    df_treino['ds'] = pd.to_datetime(df_treino['ds'])
    df_validacao['ds'] = pd.to_datetime(df_validacao['ds'])

    return df_treino, df_validacao

def treinar_e_prever(df_treino):
    modelo = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=True)
    modelo.fit(df_treino)

    futuro = modelo.make_future_dataframe(periods=8760, freq='h')
    
    previsao_completa = modelo.predict(futuro)
    
    return previsao_completa

def avaliar_modelo(df_validacao, previsao_completa):
    previsao_2025 = previsao_completa[previsao_completa['ds'] >= '2025-01-01']
    
    dados_reais = df_validacao['y']
    dados_previstos = previsao_2025['yhat']

    mae = mean_absolute_error(dados_reais, dados_previstos)
    r2 = r2_score(dados_reais, dados_previstos)

    print(f"Erro Médio Absoluto (MAE): {mae:.2f}°C")
    print(f"Coeficiente de Determinação (R²): {r2 * 100:.2f}%")
    
    return mae

def gerar_graficos(df_treino, df_validacao, previsao_completa, mae):
    df_completo = pd.concat([df_treino, df_validacao])

    plt.figure(figsize=(14, 7))
    
    plt.plot(df_completo['ds'], df_completo['y'], label='Dados Reais', color='black', marker='.', zorder=1)
    
    plt.plot(previsao_completa['ds'], previsao_completa['yhat'], label='Previsão', color='blue', linestyle='--', linewidth=2, zorder=2)
    
    plt.fill_between(previsao_completa['ds'], previsao_completa['yhat_lower'], previsao_completa['yhat_upper'], color='blue', alpha=0.2)
    
    plt.axvline(x=pd.Timestamp('2025-01-01'), color='red', linestyle='-', linewidth=2, label='Corte (2024/2025)')

    plt.title(f'Previsão Prophet | Ano de Corte: 2024 | MAE: {mae:,.2f}°C', fontsize=16, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

def main():
    df_treino, df_validacao = carregar_datasets()
    
    previsao_completa = treinar_e_prever(df_treino)
    
    mae = avaliar_modelo(df_validacao, previsao_completa)
    
    gerar_graficos(df_treino, df_validacao, previsao_completa, mae)

if __name__ == "__main__":
    main()