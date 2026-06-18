Projetos de Previsão de Séries Temporais com Prophet

Este projeto contém scripts em Python que demonstram a aplicação prática da biblioteca Prophet para a previsão de séries temporais. O objetivo é ilustrar como o algoritmo lida com diferentes frequências de dados e tipos de sazonalidade em dois cenários de negócios distintos.

Estrutura do Projeto

O projeto é dividido em dois casos de uso principais:

1. Análise de Sensores de Temperatura (IoT)

Arquivo: analise_temperatura.py

Focado na previsão de dados em alta frequência (horária). O modelo é treinado com dados históricos de temperatura de um ano base (2024) para prever o comportamento do ano seguinte (2025).

Propósito: Estabelecer uma linha de base de normalidade térmica, capturando sazonalidades diárias, semanais e anuais.

Dados: Requer os arquivos dados_temperatura_treino.csv e dados_temperatura_validacao.csv no diretório local.

2. Análise de Volume de Vendas no Varejo

Arquivo: analise_varejo.py (ou nome equivalente do segundo script)

Focado na previsão de dados macroeconômicos em baixa frequência (mensal). O script consome um dataset público e testa a capacidade preditiva do Prophet utilizando múltiplos pontos de corte histórico (2004, 2006 e 2008).

Propósito: Prever o volume de vendas a longo prazo aplicando o modo de sazonalidade multiplicativa, ideal para cenários onde a variação dos ciclos aumenta conforme o volume cresce.

Dados: Consome automaticamente o dataset example_retail_sales.csv diretamente do repositório oficial do Prophet.

Dependências

Para executar os scripts, é necessário instalar as seguintes bibliotecas Python:

pandas

prophet

matplotlib

scikit-learn

Você pode instalá-las rodando o comando:
pip install pandas prophet matplotlib scikit-learn

Como Executar

Execute qualquer um dos scripts diretamente pelo terminal. Ao final da execução, as métricas de Erro Médio Absoluto (MAE) serão calculadas e um gráfico interativo contendo os dados reais, a linha de previsão, o ponto de corte e o intervalo de confiança será exibido na tela.

Exemplo de execução:
python analise_temperatura.py
