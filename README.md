# Análise de Séries Temporais de Sensores de Temperatura

Este projeto utiliza a biblioteca Prophet para modelar e prever o comportamento térmico de ambientes com base em dados históricos de sensores (IoT). O objetivo principal é estabelecer uma linha de base de normalidade térmica, compreendendo sazonalidades diárias, semanais e anuais, o que permite a futura identificação de anomalias (como falhas em sistemas de ar-condicionado ou refrigeração).

## Estrutura do Projeto

* `analise_temperatura.py`: Script principal que carrega os dados, treina o modelo, avalia a precisão e gera os gráficos.
* `dados_temperatura_treino.csv`: Dados históricos do ano base (ex: 2024) utilizados para treinar o algoritmo.
* `dados_temperatura_validacao.csv`: Dados do ano seguinte (ex: 2025) utilizados exclusivamente para validar a precisão das previsões geradas.
* `requirements.txt`: Lista de dependências do Python necessárias para rodar o projeto.

## Resultados Gerados

Ao executar o script, os seguintes arquivos serão criados na raiz do projeto:
* `grafico1_previsao_2025.png`: Comparativo visual entre o histórico real e a projeção do modelo (incluindo as margens de confiança).
* `grafico2_dados_reais.png`: Linha do tempo completa exibindo os dados reais de ambos os anos.

## Como Executar

1. Certifique-se de ter o Python instalado em sua máquina.
2. Abra o terminal na pasta do projeto e instale as dependências:
   pip install -r requirements.txt

3. Execute o script principal:
   python analise_temperatura.py

Após a execução, os resultados das métricas (MAE e R²) serão exibidos no terminal e os gráficos serão salvos no diretório atual.
