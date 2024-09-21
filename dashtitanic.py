import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc
import plotly.figure_factory as ff

# Carregar os dados
df = pd.read_csv('train.csv')

# Pré-processamento básico
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = 0
df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
df = df.dropna(subset=['Age'])  # Remover valores nulos de Age para simplificar

# Treinar o modelo Random Forest
X_train = df[['Pclass', 'Sex', 'Age', 'Fare', 'FamilySize', 'IsAlone']]
y_train = df['Survived']
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train, y_train)

# Inicializar o app Dash com tema Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout do Dashboard
app.layout = dbc.Container([
    html.H1("Dashboard Titanic - Análise de Sobrevivência", style={'text-align': 'center', 'margin-bottom': '20px'}),

    # Dropdown para selecionar o tipo de gráfico
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(id="grafico-escolha",
                         options=[
                             {"label": "Distribuição por Sexo", "value": "sexo"},
                             {"label": "Distribuição por Classe", "value": "classe"},
                             {"label": "Distribuição por Porto de Embarque", "value": "porto"},
                             {"label": "Distribuição por Família e Sobrevivência", "value": "familia"},
                             {"label": "Matriz de Correlação", "value": "correlacao"},
                             {"label": "Matriz de Confusão - Random Forest", "value": "confusao"},
                             {"label": "Curva ROC", "value": "roc"},
                             {"label": "Importância das Features", "value": "importancia_features"}
                         ],
                         multi=False,
                         value="sexo",
                         style={'width': "100%"}
                         ),
        ], width=6),
    ], justify="center"),

    # Exibir o gráfico selecionado
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='grafico-principal', figure={}),
        ], width=12)
    ]),

    # Exibir explicação sobre o gráfico
    dbc.Row([
        dbc.Col([
            html.Div(id='grafico-explicacao', style={'text-align': 'center', 'margin': '20px'})
        ], width=12)
    ]),

    # Footer
    dbc.Row([
        dbc.Col([
            html.Footer('Desenvolvido por Pablo Bormotoff', style={'text-align': 'center', 'margin-top': '40px'})
        ])
    ])
], fluid=True)


# Callback para atualizar o gráfico principal baseado na seleção
@app.callback(
    [Output('grafico-principal', 'figure'), Output('grafico-explicacao', 'children')],
    [Input('grafico-escolha', 'value')]
)
def update_graph(option_selected):
    if option_selected == "sexo":
        fig = px.histogram(df, x="Sex", color="Survived",
                           barmode='group',
                           labels={'Sex': 'Sexo (0 = Masculino, 1 = Feminino)', 'Survived': 'Sobreviveu'},
                           title="Distribuição por Sexo e Sobrevivência")
        explicacao = """
        Este gráfico mostra a distribuição de sobreviventes e não sobreviventes de acordo com o sexo. 
        Podemos observar que o sexo feminino teve uma taxa de sobrevivência maior em relação ao sexo masculino.
        """
        
    elif option_selected == "classe":
        fig = px.histogram(df, x="Pclass", color="Survived",
                           barmode='group',
                           labels={'Pclass': 'Classe', 'Survived': 'Sobreviveu'},
                           title="Distribuição por Classe e Sobrevivência")
        explicacao = """
        Este gráfico mostra a distribuição de sobrevivência com base nas classes sociais. 
        Podemos notar que passageiros da 1ª classe têm maior chance de sobreviver do que os da 3ª classe.
        """

    elif option_selected == "porto":
        fig = px.histogram(df, x="Embarked", color="Survived",
                           barmode='group',
                           labels={'Embarked': 'Porto de Embarque', 'Survived': 'Sobreviveu'},
                           title="Distribuição por Porto de Embarque e Sobrevivência")
        explicacao = """
        Este gráfico mostra a sobrevivência em relação ao porto de embarque. 
        Passageiros que embarcaram no porto C parecem ter tido uma maior taxa de sobrevivência.
        """

    elif option_selected == "familia":
        fig = px.histogram(df, x="FamilySize", color="Survived",
                           barmode='group',
                           labels={'FamilySize': 'Tamanho da Família', 'Survived': 'Sobreviveu'},
                           title="Distribuição de Tamanho da Família e Sobrevivência")
        explicacao = """
        Este gráfico apresenta a relação entre o tamanho da família e a sobrevivência. 
        Passageiros sozinhos (FamilySize = 1) tiveram uma taxa de sobrevivência menor.
        """

    elif option_selected == "correlacao":
        # Selecionar apenas colunas numéricas
        corr_matrix = df.select_dtypes(include='number').corr()

        # Converter a matriz de correlação em uma lista para visualização adequada
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='Viridis',
            zmin=-1, zmax=1
        ))

        fig.update_layout(
            title='Matriz de Correlação',
            xaxis_nticks=36
        )

        explicacao = """
        Esta é a matriz de correlação entre as variáveis do conjunto de dados. 
        Valores próximos de 1 indicam uma forte correlação positiva, enquanto valores próximos de -1 indicam uma correlação inversa.
        """

    elif option_selected == "confusao":
        y_train_pred = random_forest.predict(X_train)
        cm = confusion_matrix(y_train, y_train_pred)
        z = cm
        x = ['Não Sobreviveu', 'Sobreviveu']
        y = ['Não Sobreviveu', 'Sobreviveu']

        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Viridis', showscale=True)
        fig.update_layout(title_text='Matriz de Confusão - Random Forest')
        explicacao = """
        A matriz de confusão exibe os verdadeiros positivos, falsos positivos, verdadeiros negativos e falsos negativos do modelo Random Forest.
        """

    elif option_selected == "roc":
        fpr, tpr, _ = roc_curve(y_train, random_forest.predict_proba(X_train)[:, 1])
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='Curva ROC'))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Aleatório', line=dict(dash='dash')))
        fig.update_layout(title=f'Curva ROC (AUC = {roc_auc:.2f})',
                          xaxis_title='Taxa de Falsos Positivos', yaxis_title='Taxa de Verdadeiros Positivos')
        explicacao = """
        A curva ROC (Receiver Operating Characteristic) avalia o desempenho do modelo ao calcular a taxa de verdadeiros e falsos positivos.
        O AUC (Área sob a Curva) indica a precisão do modelo; quanto mais próximo de 1, melhor.
        """

    elif option_selected == "importancia_features":
        importances = random_forest.feature_importances_
        feature_names = X_train.columns
        fig = px.bar(x=feature_names, y=importances, labels={'x': 'Features', 'y': 'Importância'},
                     title="Importância das Features no Modelo Random Forest")
        explicacao = """
        Este gráfico exibe a importância das variáveis no modelo Random Forest. 
        Ele nos mostra quais variáveis mais impactaram nas previsões de sobrevivência.
        """

    return fig, explicacao


# Rodar o app
if __name__ == '__main__':
    app.run_server(debug=True)