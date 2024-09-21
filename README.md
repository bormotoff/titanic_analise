
# Titanic - Análise de Sobrevivência

Este projeto utiliza o dataset clássico do Titanic disponivel na Kubblr para realizar uma análise exploratória de dados, visualizações e modelagem preditiva de sobrevivência usando técnicas de Machine Learning, visando identificar taxas de sobrevivência com diversas variaveis. 

Os dados são históricos e não tem relação com o filme ou obras ficcionais baseadas nos eventos.

# Referência

#### Fonte historica: 
https://www.noaa.gov/office-of-general-counsel/gc-international-section/rms-titanic-history-and-significance

#### Fonte original dos dados
https://www.kaggle.com/c/titanic

# Links uteis no Repositório

#### Jupyter Notebook: 
https://github.com/bormotoff/titanic_analise/blob/main/analisetitanic.ipynb

#### Analise em Python:
https://github.com/bormotoff/titanic_analise/blob/main/analisetitanic.py

#### Dashboard usando plotly em Python:
https://github.com/bormotoff/titanic_analise/blob/main/dashtitanic.py
# Documentação
## Entendimento dos dados

#### PassengerId
Um identificador único para cada passageiro no conjunto de dados.

#### Sobrevivência
Variável alvo, que indica se o passageiro sobreviveu (1) ou não (0).

#### Pclass
Classe do passageiro (1 = Primeira, 2 = Segunda, 3 = Terceira).

#### Name
Nome do passageiro.

### Sex
Sexo do passageiro (male = masculino, female = feminino).

#### Age
Idade do passageiro.

#### SibSp
Número de irmãos/cônjuges a bordo com o passageiro.

#### Parch
Número de pais/filhos a bordo com o passageiro.

#### Ticket
Número do bilhete do passageiro.
Uso no Script: Não utilizado diretamente na modelagem ou análise.

#### Fare
Tarifa paga pelo passageiro.

#### Cabin
Número da cabine do passageiro.

#### Embarked
Porto de embarque do passageiro (C = Cherbourg, Q = Queenstown, S = Southampton).


# Preparação dos dados
## Bibliotecas
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import LogisticRegression

import SVC

import confusion_matrix

## Leitura dos dados
### 1 - Extrair título do nome
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

### 2 - Lista de variáveis categóricas
categorical_vars = ['Sex', 'Pclass', 'Embarked', 'Title']

### 3 - Criar um DataFrame vazio para armazenar os resultados
summary_tables = []

for var in categorical_vars:
    counts = train[var].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    counts['Variable'] = var
    counts = counts[['Variable', 'Category', 'Count']]
    summary_tables.append(counts)

#### 4 - Loop para contar categorias em cada variável categórica:
for var in categorical_vars:
    counts = train[var].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    counts['Variable'] = var
    counts = counts[['Variable', 'Category', 'Count']]
    summary_tables.append(counts)

#### 5 - Concatenar todas as tabelas:
final_summary = pd.concat(summary_tables, ignore_index=True)

# Insights explorados

### Análise Descritiva:
Fiz distribuições categóricas como Sex, Pclass, Embarked e Title, que mostram a distribuição dessas características entre os passageiros.

### Criação de Novas Features:
Variáveis como IsAlone foram criadas para verificar se passageiros que estavam sozinhos tinham diferentes chances de sobrevivência.

### Modelagem Preditiva (Random Forest):
Divide o conjunto em features (Pclass, Sex, Age, Fare, FamilySize, IsAlone) para o treinamento do modelo e a variável-alvo foi Survived.
Um modelo de Random Forest foi treinado com 100 árvores de decisão e uma semente de aleatoriedade (random_state = 42) para garantir a reprodutibilidade. O modelo foi avaliado e utilizado para prever a sobrevivência dos passageiros no conjunto de teste.
### Análises Visuais:
Criei gráficos de distribuição para visualizar a sobrevivência com base em características como sexo, classe e porto de embarque.
E foi gerada uma matriz de correlação para analisar as inter-relações entre as variáveis numéricas do conjunto de dados e uma Avaliação do desempenho do modelo através de uma matriz de confusão para verificar verdadeiros e falsos positivos/negativos.

A curva ROC avaliou o desempenho do modelo com relação à taxa de verdadeiros e falsos positivos

### Resultado

Sobrevivência por Sexo: 
Mulheres tinham uma chance significativamente maior de sobreviver em comparação aos homens.


Sobrevivência por Classe: 
Passageiros da 1ª classe tinham maior probabilidade de sobreviver, enquanto os da 3ª classe enfrentaram maiores riscos.

Sobrevivência por Porto de Embarque: 
Passageiros que embarcaram no porto C (Cherbourg) tiveram uma taxa de sobrevivência maior em relação aos outros portos.

Impacto do Tamanho da Família: 
Passageiros que estavam sozinhos tinham uma taxa de sobrevivência menor em comparação àqueles que viajavam com familiares.


O modelo Random Forest foi capaz de prever a sobrevivência dos passageiros com boa precisão. Além disso, insights importantes sobre os fatores que influenciam a sobrevivência, como sexo, classe social, e o fato de estar viajando sozinho, foram gerados a partir das análises exploratórias.
