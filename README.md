# Previsão de Séries Temporais com Prophet

Scripts em Python para previsão de séries temporais com a biblioteca Prophet, aplicados a dois cenários de negócios com frequências e tipos de sazonalidade distintos.

---

## Estrutura do Projeto

### 1. Análise de Sensores de Temperatura (IoT)

**Arquivo:** `analise_temperatura.py`

Previsão de dados em alta frequência (horária). O modelo é treinado com dados de 2024 para prever o comportamento de 2025, capturando sazonalidades diárias, semanais e anuais.

Requer os arquivos `dados_temperatura_treino.csv` e `dados_temperatura_validacao.csv` no diretório local.

### 2. Análise de Volume de Vendas no Varejo

**Arquivo:** `analise_varejo.py`

Previsão de dados em baixa frequência (mensal). Testa a capacidade preditiva do Prophet com múltiplos pontos de corte histórico (2004, 2006 e 2008), usando sazonalidade multiplicativa.

Os dados são consumidos automaticamente via `example_retail_sales.csv` do repositório oficial do Prophet.

---

## Dependências

```bash
pip install pandas prophet matplotlib scikit-learn
```

---

## Como Executar

```bash
python analise_temperatura.py
```

```bash
python analise_varejo.py
```

Ao final da execução, as métricas de Erro Médio Absoluto (MAE) são exibidas e um gráfico interativo é gerado com os dados reais, a linha de previsão, o ponto de corte e o intervalo de confiança.
