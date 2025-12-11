# 🧠 Análise Preditiva de Ansiedade com Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-yellow.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green.svg)](https://xgboost.ai/)

## 📋 Sobre o Projeto

Este projeto implementa um sistema completo de análise preditiva para identificação de ansiedade utilizando técnicas avançadas de Machine Learning. O trabalho abrange desde o pré-processamento de dados até a explicabilidade de modelos, com foco em métodos rigorosos de validação e otimização.

### 🎯 Objetivos

- Desenvolver modelos preditivos robustos para identificação de ansiedade
- Comparar diferentes algoritmos de classificação
- Otimizar hiperparâmetros através de busca bayesiana
- Fornecer interpretabilidade dos modelos através de análise SHAP
- Garantir reprodutibilidade e validação rigorosa dos resultados

## ✨ Características Principais

- **Múltiplos Algoritmos**: Regressão Logística, Random Forest e XGBoost
- **Otimização Avançada**: Utilização do Optuna com TPE Sampler para busca bayesiana de hiperparâmetros
- **Validação Cruzada Estratificada**: 5-fold cross-validation para avaliação robusta
- **Explicabilidade**: Análise SHAP completa com summary plots e dependence plots
- **Métricas Abrangentes**: AUC-ROC, Acurácia, Sensibilidade, Especificidade, Precisão e F1-Score
- **Reprodutibilidade**: Seeds fixadas e pipeline padronizado

## 🗂️ Estrutura do Projeto

```
ArtigoML/
│
├── ansiedade.ipynb                    # Notebook principal com toda a análise
├── banco_de_dados_20250720.csv        # Dataset de treinamento
├── requirements.txt                   # Dependências do projeto
└── README.md                          # Documentação do projeto
```

## 📊 Dataset

O dataset contém informações demográficas, comportamentais e de uso de redes sociais, incluindo:

- **Variáveis Contínuas**: Idade, horas de uso diário, tempo de sessões, etc.
- **Variáveis Categóricas**: Sexo, escolaridade, status de relacionamento, uso de aplicativos, etc.
- **Variável Target**: `Target_Ansioso` (0 = Não Ansioso, 1 = Ansioso)

### Pré-processamento Implementado

- Cálculo automático de idade a partir da data de nascimento
- Imputação de valores ausentes (mediana para contínuas, moda para categóricas)
- One-Hot Encoding para variáveis categóricas
- Padronização de features (StandardScaler)

## 🚀 Instalação

### Pré-requisitos

- Python 3.8 ou superior
- Jupyter Notebook ou JupyterLab

### Passos de Instalação

1. **Clone o repositório** (ou faça download dos arquivos):
```bash
git clone <url-do-repositorio>
cd ArtigoML
```

2. **Crie um ambiente virtual** (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

## 💻 Uso

### Executando o Notebook

1. **Inicie o Jupyter Notebook**:
```bash
jupyter notebook
```

2. **Abra o arquivo** `ansiedade.ipynb`

3. **Execute as células sequencialmente** para:
   - Carregar e pré-processar os dados
   - Treinar e otimizar os modelos
   - Avaliar o desempenho
   - Gerar análises de explicabilidade

### Estrutura do Notebook

1. **Importação de Bibliotecas**: Todas as dependências necessárias
2. **Carregamento e Pré-processamento**: Limpeza e transformação dos dados
3. **Treinamento e Avaliação**: 
   - Otimização com Optuna (50 trials por modelo)
   - Validação cruzada estratificada
   - Avaliação no conjunto de teste
4. **Análise de Importância**: Feature importance para Random Forest e XGBoost
5. **Explicabilidade SHAP**:
   - Geração dos valores SHAP
   - Dependence plots com interações
   - Summary plots para todos os modelos

## 📈 Resultados

### Modelos Implementados

| Modelo | Descrição |
|--------|-----------|
| **Regressão Logística** | Modelo linear com regularização L1/L2 otimizada |
| **Random Forest** | Ensemble de árvores de decisão com 50-300 estimadores |
| **XGBoost** | Gradient boosting otimizado para classificação |

### Métricas de Avaliação

O projeto calcula as seguintes métricas no conjunto de teste:

- **AUC-ROC**: Área sob a curva ROC (métrica principal)
- **Acurácia**: Proporção de predições corretas
- **Sensibilidade (Recall)**: Taxa de verdadeiros positivos
- **Especificidade**: Taxa de verdadeiros negativos
- **Precisão (PPV)**: Valor preditivo positivo
- **F1-Score**: Média harmônica de precisão e recall

### Explicabilidade

- **SHAP Values**: Quantificação da contribuição de cada feature
- **Dependence Plots**: Relação entre features e predições com interações
- **Summary Plots**: Visão geral da importância e distribuição dos valores SHAP

## 🛠️ Tecnologias Utilizadas

### Core
- **Python 3.8+**: Linguagem de programação
- **Pandas**: Manipulação de dados
- **NumPy**: Operações numéricas

### Machine Learning
- **scikit-learn**: Algoritmos de ML e métricas
- **XGBoost**: Gradient boosting otimizado
- **Optuna**: Otimização bayesiana de hiperparâmetros

### Interpretabilidade
- **SHAP**: Análise de explicabilidade de modelos

### Visualização
- **Matplotlib**: Criação de gráficos

## 📝 Metodologia

### 1. Divisão de Dados
- 80% treino / 20% teste
- Estratificação para manter proporção das classes

### 2. Otimização de Hiperparâmetros
- Algoritmo: TPE (Tree-structured Parzen Estimator)
- Trials: 50 iterações por modelo
- Métrica de otimização: AUC-ROC
- Validação: 5-fold cross-validation estratificada

### 3. Avaliação Final
- Treinamento com hiperparâmetros otimizados
- Avaliação no conjunto de teste não visto
- Cálculo de múltiplas métricas de desempenho

### 4. Interpretabilidade
- Análise SHAP para os três modelos
- Identificação das top 5 features mais importantes
- Visualização de interações entre features

## 🔬 Reprodutibilidade

O projeto implementa várias medidas para garantir reprodutibilidade:

- `SEED = 42` fixada em todos os geradores aleatórios
- `random_state` definido em todos os modelos
- `n_jobs=1` durante otimização para consistência
- Pipeline padronizado para pré-processamento

## 📄 Licença

Este projeto é disponibilizado para fins acadêmicos e de pesquisa.

## 👥 Contribuindo

Contribuições são bem-vindas! Para contribuir:

1. Faça um fork do projeto
2. Crie uma branch para sua feature (`git checkout -b feature/NovaFeature`)
3. Commit suas mudanças (`git commit -m 'Adiciona nova feature'`)
4. Push para a branch (`git push origin feature/NovaFeature`)
5. Abra um Pull Request

## 📧 Contato

Para questões ou sugestões sobre o projeto, por favor abra uma issue no repositório.

---

**Nota**: Este projeto utiliza dados sensíveis de saúde mental. Certifique-se de seguir todas as diretrizes éticas e de privacidade ao trabalhar com os dados.
