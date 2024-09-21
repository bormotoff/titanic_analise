import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configurar estilo dos gráficos
sns.set(style='whitegrid', palette='muted')

# Carregar os dados
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Informações sobre os dados
train.info()

# Visualizar as primeiras linhas do DataFrame (opcional)
print(train.head())

# Extrair título do nome
train['Title'] = train['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Lista de variáveis categóricas
categorical_vars = ['Sex', 'Pclass', 'Embarked', 'Title']

# Criar um DataFrame vazio para armazenar os resultados
summary_tables = []

for var in categorical_vars:
    counts = train[var].value_counts().reset_index()
    counts.columns = ['Category', 'Count']
    counts['Variable'] = var
    counts = counts[['Variable', 'Category', 'Count']]
    summary_tables.append(counts)

# Concatenar todas as tabelas
final_summary = pd.concat(summary_tables, ignore_index=True)

# Exibir a tabela final
print("\nTabela Resumida das Variáveis Categóricas:")
print(final_summary)

# **Visualização**

# Gráfico de Distribuição por Sexo com Sobrevivência
plt.figure(figsize=(6, 4))
sns.countplot(x='Sex', hue='Survived', data=train)
plt.title('Distribuição por Sexo e Sobrevivência')
plt.xlabel('Sexo')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.savefig('Distribuicao_Sexo_Sobrevivencia.png')
plt.show()

# Gráfico de Distribuição por Classe com Sobrevivência
plt.figure(figsize=(6, 4))
sns.countplot(x='Pclass', hue='Survived', data=train)
plt.title('Distribuição por Classe e Sobrevivência')
plt.xlabel('Classe')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.savefig('Distribuicao_Classe_Sobrevivencia.png')
plt.show()

# Gráfico de Distribuição por Porto de Embarque com Sobrevivência
plt.figure(figsize=(6, 4))
sns.countplot(x='Embarked', hue='Survived', data=train)
plt.title('Distribuição por Porto de Embarque e Sobrevivência')
plt.xlabel('Porto de Embarque')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.savefig('Distribuicao_Porto_Embarque_Sobrevivencia.png')
plt.show()

# Gráfico de Distribuição por Título com Sobrevivência
plt.figure(figsize=(10, 6))
sns.countplot(y='Title', hue='Survived', data=train, order=train['Title'].value_counts().index)
plt.title('Distribuição por Título e Sobrevivência')
plt.xlabel('Contagem')
plt.ylabel('Título')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.savefig('Distribuicao_Titulo_Sobrevivencia.png')
plt.show()

# Pré-processamento dos Dados

# Imputar idade com a mediana
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

# Imputar Embarked com a moda
train['Embarked'] = train['Embarked'].fillna(train['Embarked'].mode()[0])

# Imputar Fare no conjunto de teste
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

# Converter sexo em numérico
train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

# Criar dummies para Embarked
embarked_dummies_train = pd.get_dummies(train['Embarked'], prefix='Embarked')
embarked_dummies_test = pd.get_dummies(test['Embarked'], prefix='Embarked')

train = pd.concat([train, embarked_dummies_train], axis=1)
test = pd.concat([test, embarked_dummies_test], axis=1)

# Criar nova feature FamilySize
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1

# Criar coluna IsAlone usando .loc para evitar o warning
# Para o conjunto de treinamento
train['IsAlone'] = 1  # Assume inicialmente que todos estão sozinhos
train.loc[train['FamilySize'] > 1, 'IsAlone'] = 0  # Se FamilySize > 1, não está sozinho

# Para o conjunto de teste
test['IsAlone'] = 1
test.loc[test['FamilySize'] > 1, 'IsAlone'] = 0

# Extrair título do nome (já extraído anteriormente para train)
test['Title'] = test['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)

# Mapear títulos raros
title_mapping = {
    'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Dr': 5,
    'Rev': 5, 'Col': 5, 'Major': 5, 'Mlle': 2, 'Countess': 5,
    'Ms': 2, 'Lady': 5, 'Jonkheer': 5, 'Don': 5, 'Dona': 5,
    'Mme': 3, 'Capt': 5, 'Sir': 5
}
train['Title'] = train['Title'].map(title_mapping)
test['Title'] = test['Title'].map(title_mapping)

# Preencher títulos nulos com 0
train['Title'] = train['Title'].fillna(0)
test['Title'] = test['Title'].fillna(0)

# **Análises e Gráficos Adicionais**

# 1. Matriz de Correlação
numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'Pclass', 'Sex', 'Title', 'IsAlone']
corr_df = train[numeric_features + ['Survived']]
corr_matrix = corr_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matriz de Correlação das Variáveis')
plt.savefig('Matriz_correlacao_Variaveis.png')
plt.show()

# 2. Sobrevivência por Classe (já foi plotado anteriormente)

# 3. Sobrevivência por Porto de Embarque (já foi plotado anteriormente)

# 4. Distribuição de Fare por Sobrevivência
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Fare', data=train)
plt.title('Distribuição de Tarifa por Sobrevivência')
plt.xlabel('Sobreviveu')
plt.ylabel('Tarifa')
plt.xticks([0, 1], ['Não', 'Sim'])
plt.savefig('Fare_sobrevivencia.png')
plt.show()

# 5. Distribuição de Idade por Sobrevivência
plt.figure(figsize=(8, 6))
sns.boxplot(x='Survived', y='Age', data=train)
plt.title('Distribuição de Idade por Sobrevivência')
plt.xlabel('Sobreviveu')
plt.ylabel('Idade')
plt.xticks([0, 1], ['Não', 'Sim'])
plt.savefig('Idade_sobrevivencia.png')
plt.show()

# 6. Sobrevivência por Estar Sozinho ou Acompanhado
plt.figure(figsize=(8, 6))
sns.countplot(x='IsAlone', hue='Survived', data=train)
plt.title('Sobrevivência por Estar Sozinho ou Acompanhado')
plt.xlabel('Está Sozinho (1 = Sim, 0 = Não)')
plt.ylabel('Contagem')
plt.legend(title='Sobreviveu', labels=['Não', 'Sim'])
plt.savefig('Sozinho_acompanhado.png')
plt.show()

# Gráfico de contagem de sobreviventes
plt.figure(figsize=(8, 6))
sns.countplot(x='Survived', data=train)
plt.title('Distribuição de Sobrevivência')
plt.xlabel('Sobreviveu')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Não', 'Sim'])
plt.savefig('Contagem_sobreviventes.png')
plt.show()

# Sobrevivência por sexo (já foi plotado anteriormente)

# Histograma da idade por sobrevivência
g = sns.FacetGrid(train, col='Survived', height=5)
g.map(plt.hist, 'Age', bins=20)
g.set_axis_labels('Idade', 'Frequência')
g.add_legend()
plt.savefig('Histograma_idade_sobrevivencia.png')
plt.show()

# **Modelagem**

# Selecionar features
features = ['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'Title',
            'Embarked_C', 'Embarked_Q', 'Embarked_S', 'IsAlone']
X = train[features]
y = train['Survived']
X_test = test[features]

# **Dividir o conjunto de dados em treinamento e validação**
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# **Treinar o modelo no conjunto de treinamento**
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# **Fazer previsões no conjunto de validação**
y_val_pred_proba = random_forest.predict_proba(X_val)[:, 1]

# **Calcular a curva ROC**
from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
roc_auc = auc(fpr, tpr)

# **Plotar a curva ROC**
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (AUC = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falso Positivo')
plt.ylabel('Taxa de Verdadeiro Positivo')
plt.title('Curva ROC - Random Forest')
plt.legend(loc="lower right")
plt.savefig('Curva_ROC_RF.png')
plt.show()

# **Avaliar a acurácia no conjunto de validação**
from sklearn.metrics import accuracy_score

y_val_pred = random_forest.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Acurácia no Conjunto de Validação: {accuracy:.4f}')

# **Análise de Importância das Features**
importances = random_forest.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Gráfico de Importância das Features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Importância das Features no Modelo Random Forest')
plt.xlabel('Importância')
plt.ylabel('Feature')
plt.savefig('Importancia_das_Features.png')
plt.show()

# **Matriz de Confusão**
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Não Sobreviveu', 'Sobreviveu'])
disp.plot()
plt.title('Matriz de Confusão - Random Forest')
plt.savefig('Matriz_Confusao_RF.png')
plt.show()

# **Treinar o modelo final no conjunto completo**
random_forest.fit(X, y)

# **Prever no conjunto de teste**
y_test_pred = random_forest.predict(X_test)

# Criar DataFrame com as previsões
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': y_test_pred
})

# Salvar em CSV
submission.to_csv('submission.csv', index=False)

# **Interação entre Sexo e Classe**
# Gráfico de interação entre Sexo e Classe
plt.figure(figsize=(8, 6))
sns.pointplot(x='Pclass', y='Survived', hue='Sex', data=train,
              markers=['o', 'x'], linestyles=['-', '--'])
plt.title('Taxa de Sobrevivência por Classe e Sexo')
plt.xlabel('Classe')
plt.ylabel('Taxa de Sobrevivência')
plt.legend(title='Sexo', labels=['Feminino', 'Masculino'])
plt.savefig('Interacao_Sexo_Classe.png')
plt.show()
