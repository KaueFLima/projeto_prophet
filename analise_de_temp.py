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
    previsao_2025 = previsao_completa[previsao_completa['ds'] >= '2025-01-01'].reset_index(drop=True)
    
    return previsao_2025

def avaliar_modelo(df_validacao, previsao_2025):
    dados_reais = df_validacao['y']
    dados_previstos = previsao_2025['yhat']

    mae = mean_absolute_error(dados_reais, dados_previstos)
    r2 = r2_score(dados_reais, dados_previstos)

    print(f"Erro Médio Absoluto (MAE): {mae:.2f}°C")
    print(f"Coeficiente de Determinação (R²): {r2 * 100:.2f}%")

def gerar_graficos(df_treino, df_validacao, previsao_2025):
    # Grafico com os dados pela metade + previsão
    plt.figure(figsize=(14, 6))
    plt.plot(df_treino['ds'], df_treino['y'], color='royalblue', label='Real 2024 (Treino)', alpha=0.6, linewidth=1)
    plt.plot(previsao_2025['ds'], previsao_2025['yhat'], color='darkorange', label='Previsão 2025 (Prophet)', alpha=0.8, linewidth=1.2)
    plt.fill_between(previsao_2025['ds'], previsao_2025['yhat_lower'], previsao_2025['yhat_upper'], color='orange', alpha=0.3, label='Intervalo de Confiança')
    plt.title('Cenário 1: Histórico Real de 2024 e Projeção para 2025', fontsize=14, pad=15)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('grafico1_previsao_2025.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Grafico com os dados completos
    plt.figure(figsize=(14, 6))
    plt.plot(df_treino['ds'], df_treino['y'], color='royalblue', label='Real 2024', alpha=0.6, linewidth=1)
    plt.plot(df_validacao['ds'], df_validacao['y'], color='forestgreen', label='Real 2025 (Validação)', alpha=0.6, linewidth=1)
    plt.title('Cenário 2: Linha do Tempo Completa com Dados Reais (2024 e 2025)', fontsize=14, pad=15)
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Temperatura (°C)', fontsize=12)
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('grafico2_dados_reais.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    df_treino, df_validacao = carregar_datasets()
    
    previsao_2025 = treinar_e_prever(df_treino)
    
    avaliar_modelo(df_validacao, previsao_2025)
    
    gerar_graficos(df_treino, df_validacao, previsao_2025)

if __name__ == "__main__":
    main()
