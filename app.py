import dash
from dash import dcc
from dash import html
from dash import dash_table

import plotly.express as px
import matplotlib.pyplot as plt

import plotly.tools
import pandas as pd
from sklearn import svm, tree, linear_model, neighbors
from sklearn import naive_bayes, ensemble, discriminant_analysis, gaussian_process
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
from sklearn.metrics import auc, roc_auc_score, roc_curve
from dash.dependencies import Input, Output
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import svm

#leemos los datos
attrition = pd.read_csv('/Users/daritmartinez/dash_project/trabajofinal/data/WA_Fn-UseC_-HR-Employee-Attrition.csv')

plot_df = attrition.groupby(['Gender'])['Attrition'].value_counts(normalize=True)
plot_df = plot_df.mul(100).rename('Percent').reset_index()
figsex = px.bar(plot_df, x="Gender", y="Percent", color="Attrition", barmode="group",
            text='Percent', opacity=.75, category_orders={'Attrition': ['Yes', 'No']},
            color_discrete_map={'Yes': 'Green','No': 'Yellow'}) 
figsex.update_traces(texttemplate='%{text:.3s}%', textposition='outside',
                  marker_line=dict(width=1, color='#28221D'),  width=.4)
figsex.update_layout(title_text='Tasa de rotación según género', yaxis_ticksuffix = '%',
                  paper_bgcolor='#F4F2F0', plot_bgcolor='#F4F2F0',font_color='#28221D',
                  height=500, xaxis=dict(tickangle=30), )
figsex.update_xaxes(showticklabels=True,tickangle=30,col=2)
figsex.update_yaxes(title = "", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
figsex.update_layout(xaxis_title="Género", yaxis_title="Porcentaje")

cat_cols=attrition.select_dtypes(include=object).columns.tolist()
cat_df=pd.DataFrame(attrition[cat_cols].melt(var_name='column', value_name='value')
                    .value_counts()).rename(columns={0: 'count'}).sort_values(by=['column', 'count'])

attrition['AgeGroup']=pd.cut(attrition['Age'],bins=[15,20,25,30,35,40,45,50,55,60])
attrition['AgeGroup'] = attrition['AgeGroup'].astype('str')
cat_df=pd.DataFrame(attrition[cat_cols].melt(var_name='column', value_name='value')
                    .value_counts()).rename(columns={0: 'count'}).sort_values(by=['column', 'count'])
plot_df = attrition.groupby(['Gender','AgeGroup'])['Attrition'].value_counts(normalize=True)
plot_df = plot_df.mul(100).rename('Percent').reset_index()
figage = px.bar(plot_df, x="AgeGroup", y="Percent", color="Attrition", barmode="group",
            text='Percent', opacity=.75, facet_col="Gender", category_orders={'Attrition': ['Yes', 'No']},
            color_discrete_map={'Yes': '#C02B34','No': '#CDBBA7'}) 
figage.update_traces(texttemplate='%{text:.3s}%', textposition='outside',
                  marker_line=dict(width=1, color='#28221D'),  width=.4)
figage.update_layout(title_text='Tasa de rotación por grupo de edad y género', yaxis_ticksuffix = '%',
                  paper_bgcolor='#F4F2F0', plot_bgcolor='#F4F2F0',font_color='#28221D',
                  height=500, xaxis=dict(tickangle=30))
figage.update_xaxes(showticklabels=True,tickangle=30,col=2)
figage.update_yaxes(title = "", zeroline=True, zerolinewidth=1, zerolinecolor='#28221D')
figage.update_layout(xaxis_title="Grupo etáreo")


plot_df = attrition.copy()
plot_df['JobLevel'] = pd.Categorical(
    plot_df['JobLevel']).rename_categories( 
    ['Entry level', 'Mid level', 'Senior', 'Lead', 'Executive'])
col=['#73AF8E', '#4F909B', '#707BAD', '#A89DB7','#C99193']
figsal = px.scatter(plot_df, x='TotalWorkingYears', y='MonthlyIncome', facet_col="Attrition",
                 color='JobLevel',
                 color_discrete_sequence=col, 
                 category_orders={'JobLevel': ['Entry level', 'Mid level', 'Senior', 'Lead', 'Executive']})
figsal.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                  title='Rotación en función de la experiencia laboral y el salario<br>',
                  yaxis=dict(title='Income',tickprefix='$'), 
                  legend_title='', font_color='#28221D',
                  margin=dict(l=40, r=30, b=80, t=120),paper_bgcolor='#F4F2F0', plot_bgcolor='#F4F2F0')
figsal.update_layout(xaxis_title="Tiempo de experiencia", yaxis_title="Salario")

import plotly.graph_objects as go
figgen=go.Figure()
plot_df2=attrition[attrition.Attrition=='Yes']
plot_df2=plot_df2['Gender'].value_counts(normalize=True)
plot_df2=plot_df2.mul(100).rename('Percent').reset_index().sort_values('Percent', ascending=False)
plot_df2.rename(columns={'index':'Gender'}, inplace=True)
figgen.add_trace(go.Pie(labels=plot_df2['Gender'], values=plot_df2['Percent'], opacity=0.85, hole=0.4,
                     hovertemplate='%{label}<br>Attrition Rate: %{value:.3}%<extra></extra>',
                     marker_colors=['#7EB6FF','#FFA6CC']))
figgen.update_yaxes(tickmode = 'array', range=[0, 90], dtick=5)
figgen.update_traces(textfont_size=14,textfont_color='black',marker=dict(line=dict(color='#28221D', width=1)))
figgen.update_layout(title_text="Distribución por género", font_color='#28221D',
                  paper_bgcolor='#F4F2F0', plot_bgcolor='#F4F2F0')

figsalsex=go.Figure()
colors=['#FFA6CC','#7EB6FF']
for i, j in enumerate(attrition['Gender'].unique()):
    df_plot=attrition[attrition['Gender']==j]
    figsalsex.add_trace(go.Box(x=df_plot['Gender'], y=df_plot['MonthlyIncome'],
                         notched=True, line=dict(color=colors[i]),name=j))
figsalsex.update_layout(title='Salario por género',
                  boxmode='group', font_color='#28221D',
                  xaxis = dict(tickmode = 'array', tickvals = [1, 2, 3, 4],),
                  paper_bgcolor='#F4F2F0', plot_bgcolor='#F4F2F0')
                  
figsalsex.update_layout(xaxis_title="Género", yaxis_title="Salario")

# Create a label encoder object
le = LabelEncoder()

le_count = 0
for col in attrition.columns[1:]:
    if attrition[col].dtype == 'object':
        if len(list(attrition[col].unique())) <= 2:
            le.fit(attrition[col])
            attrition[col] = le.transform(attrition[col])
            le_count += 1

# convert rest of categorical variable into dummy
attrition = pd.get_dummies(attrition, drop_first=True)

# import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 5))
attrition_col = list(attrition.columns)
attrition_col.remove('Attrition')
for col in attrition_col:
    attrition[col] = attrition[col].astype(float)
    attrition[[col]] = scaler.fit_transform(attrition[[col]])
attrition['Attrition'] = pd.to_numeric(attrition['Attrition'], downcast='float')
attrition.head()

names1 = ['No', 'Si']
fig = px.pie(values=attrition.Attrition.value_counts(), names=names1)
fig.update_layout(title="Tasa de rotación")


target = attrition['Attrition'].copy()
# let's remove the target feature and redundant features from the dataset
attrition.drop(['Attrition', 'EmployeeCount', 'EmployeeNumber',
            'StandardHours', 'Over18'], axis=1, inplace=True)

# dividiendo la base de datos
X_train, X_test, y_train, y_test = train_test_split(attrition,
                                                    target,
                                                    test_size=0.25,
                                                    random_state=7,
                                                    stratify=target) 



# selection of algorithms to consider and set performance measure
models = []
models.append(('Logistic Regression', LogisticRegression(solver='liblinear', random_state=7,
                                                         class_weight='balanced')))
models.append(('Random Forest', RandomForestClassifier(
    n_estimators=100, random_state=7)))
models.append(('SVM', SVC(gamma='auto', random_state=7)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('Decision Tree Classifier',
               DecisionTreeClassifier(random_state=7)))
models.append(('Gaussian NB', GaussianNB()))


acc_results = []
auc_results = []
names = []
# set table to table to populate with performance results
col = ['Algorithm', 'ROC AUC Mean', 'ROC AUC STD', 
       'Accuracy Mean', 'Accuracy STD']
attrition_results = pd.DataFrame(columns=col)
i = 0
# evaluate each model using cross-validation
for name, model in models:
    kfold = model_selection.KFold(
        n_splits=10)  # 10-fold cross-validation

    cv_acc_results = model_selection.cross_val_score(  # accuracy scoring
        model, X_train, y_train, cv=kfold, scoring='accuracy')

    cv_auc_results = model_selection.cross_val_score(  # roc_auc scoring
        model, X_train, y_train, cv=kfold, scoring='roc_auc')

    acc_results.append(cv_acc_results)
    auc_results.append(cv_auc_results)
    names.append(name)
    attrition_results.loc[i] = [name,
                         round(cv_auc_results.mean()*100, 2),
                         round(cv_auc_results.std()*100, 2),
                         round(cv_acc_results.mean()*100, 2),
                         round(cv_acc_results.std()*100, 2)
                         ]
    i += 1
attrition_results.sort_values(by=['ROC AUC Mean'], ascending=False)


def repetir(lista, veces):
    salida = []
    for elemento in lista:
        salida.extend([elemento] * veces)
    return salida

import plotly.express as px
df = pd.DataFrame(np.concatenate((auc_results)))
df['Algoritmo']=repetir(names,10)
figbox = px.box(df, x=df.Algoritmo, y=0,
                 width=750, height=400)








def generate_table(dataframe):
    return html.Table([ 
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), 6))
        ])
    ])


# LogisticRegression
param_grid = {'C': np.arange(0.1, 2, 0.1)} # hyper-parameter list to fine-tune
log_gs = GridSearchCV(LogisticRegression(solver='liblinear', # setting GridSearchCV
                                         class_weight="balanced", 
                                         random_state=7),
                      #iid=True,
                      return_train_score=True,
                      param_grid=param_grid,
                      scoring='roc_auc',
                      cv=10)
log_grid = log_gs.fit(X_train, y_train)
log_opt = log_grid.best_estimator_

fpr, tpr, thresholds = roc_curve(y_test, log_opt.predict_proba(X_test)[:,1])
figlogreg = px.line(x=fpr, y=tpr, title='Logistic Regression')
figlogreg.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

# RandomForestClassifier
rf_classifier = RandomForestClassifier(class_weight = "balanced",
                                       random_state=7)
param_grid = {'n_estimators': [100],
              'min_samples_split':[8],
              'min_samples_leaf': [3],
              'max_depth': [15]}

rf_obj = GridSearchCV(rf_classifier,
                        #iid=True,
                        return_train_score=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=10)

grid_fit = rf_obj.fit(X_train, y_train)
rf_opt = grid_fit.best_estimator_

rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_opt.predict_proba(X_test)[:,1])
figrf = px.line(x=rf_fpr, y=rf_tpr, title='Random Forest')
figrf.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")

# SMV

smv = SVC(gamma='auto', random_state=7)
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']}
smv_obj =  GridSearchCV(smv,
                        #iid=True,
                        param_grid=param_grid,
                        scoring='roc_auc',cv=5)
grid_fit = smv_obj.fit(X_train, y_train)
smv_opt = grid_fit.best_estimator_

smv_fpr, smv_tpr, smv_thresholds = roc_curve(y_test, smv_opt.decision_function(X_test))
figsmv = px.line(x=smv_fpr, y=smv_tpr, title='Soporte Maquina Vectorial')
figsmv.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")



# KNN

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': [3, 4, 5]} # hyper-parameter list to fine-tune
knn_obj =  GridSearchCV(knn,
                        #iid=True,
                        param_grid=param_grid,
                        scoring='roc_auc')
knn_fit = knn_obj.fit(X_train, y_train)
knn_opt = knn_fit.best_estimator_

knn_fpr, knn_tpr, knn_thresholds = roc_curve(y_test, knn_opt.predict_proba(X_test)[:,1])
figknn = px.line(x=knn_fpr, y=knn_tpr, title='KNN')
figknn.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")


# Decision Tree Classifier

dtc = DecisionTreeClassifier(random_state=7)
param_grid = {'criterion':['gini','entropy'],'max_depth':[4,5,6,7,8,9,10,11,12,15,20,30,40,50,70,90,120,150]}
dtc_obj =  GridSearchCV(dtc,
                        #iid=True,
                        param_grid=param_grid,
                        scoring='roc_auc',
                        cv=5)
dtc_fit = dtc_obj.fit(X_train, y_train)
dtc_opt = dtc_fit.best_estimator_

dtc_fpr, dtc_tpr, dtc_thresholds = roc_curve(y_test, dtc_opt.predict_proba(X_test)[:,1])
figdtc = px.line(x=dtc_fpr, y=dtc_tpr, title='Decision Tree Classifier')
figdtc.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")


# GaussianNB

nb_classifier = GaussianNB()

param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}
gs_NB = GridSearchCV(nb_classifier, 
                 param_grid=param_grid, 
                 cv=5,   # use any cross validation technique 
                 verbose=1, 
                 scoring='roc_auc') 
gs_NB_fit = gs_NB.fit(X_train, y_train)
gs_NB_opt = gs_NB_fit.best_estimator_

gs_fpr, gs_tpr, gs_thresholds = roc_curve(y_test, gs_NB_opt.predict_proba(X_test)[:,1])
figgs = px.line(x=gs_fpr, y=gs_tpr, title='Gaussian NB')
figgs.update_layout(xaxis_title="Tasa de falsos positivos", yaxis_title="Tasa de verdaderos positivos")


def actualizar_grafico(modelo):
    figura = ""
    if modelo == 'Logistic Regression':
        
        figura = figlogreg
        
    elif modelo == 'Random Forest':
        figura = figrf
    elif modelo == 'KNN':
        figura = figknn
    elif modelo == 'Decision Tree Classifier':
        figura = figdtc
    elif modelo == 'Gaussian NB':
        figura = figgs
    elif modelo == 'SVM':
        figura = figsmv
    return figura

def generate_table(dataframe):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), 6))
        ])
    ])


def comparate_models(model1, model2):
    log1 = ""
    log2 = ""
    if model1 == 'Logistic Regression':
        log1 = log_gs
    elif model1 == 'Random Forest':
        log1 = rf_obj
    elif model1 == 'KNN':
        log1 = knn_fit
    elif model1 == 'Decision Tree Classifier':
        log1 = dtc_fit
    elif model1 == 'Gaussian NB':
        log1 = gs_NB
    elif model1 == 'SVM':
        log1 = smv_obj

    if model2 == 'Logistic Regression':
        log2 = log_gs
    elif model2 == 'Random Forest':
        log2 = rf_obj
    elif model2 == 'KNN':
        log2 = knn_fit
    elif model2 == 'Decision Tree Classifier':
        log2 = dtc_fit
    elif model2 == 'Gaussian NB':
        log2 = gs_NB
    elif model2 == 'SVM':
        log2 = smv_obj

        log1.update_layout(xaxis_title="X Axis Title", yaxis_title="X Axis Title")

    return html.Div([
                    html.H3(model1),
                    html.H4("Mejores parametros: {}".format(log1.best_params_)),
                    html.H4("Mejor Score: {}".format(round(log1.best_score_,2))),
                    html.H3(model2),
                    html.H4("Mejores parametros: {}".format(log2.best_params_)),
                    html.H4("Mejor Score: {}".format(round(log2.best_score_,2)))
                    ], style={'columnCount': 2})



app = dash.Dash(__name__,suppress_callback_exceptions=True)

app.layout = html.Div(children=[
     
    html.H1('Modelo de rotación de empleados con una retención estratégica'),

    dcc.Tabs(id="tabs-graph", value='tab-1-graph', children=[
        dcc.Tab(label='Análisis Descriptivo', value='tab-1-graph'),
        dcc.Tab(label='Predicciones', value='tab-2-graph'),
    ]),

    html.Div(id='tabs-content-graph'),
   
    html.Div(id='tabs-content2-graph'),

    html.Div(id='display-selected-values')

  
])





@app.callback(Output('tabs-content-graph', 'children'),
              Input('tabs-graph', 'value'))
def render_content(tab):
    if tab == 'tab-1-graph':
        return html.Div([
            dcc.Markdown('''
            ## Introducción

            De acuerdo con Darapaneni et al., 2022,uno de los principales problemas actuales de la mayoría de organizaciones a nivel global 
            es la rotación de sus colaboradores. El autor refiere que una organización en promedio experimenta deserciones entre un 12 y 15%, 
            generando incremento en los gastos para vincular nuevos colaboradores que satisfagan sus necesidades, con el agravante de la 
            disminución de la producción mientras los nuevos colaboradores se adaptan al ritmo que las organizaciones requieren. Por su parte 
            Rey Caldeyro (2021) refiere que en Estados Unidos en 2019 más de un 27% de los empleados renunciaron voluntariamente a sus puestos 
            de trabajo y se estima que para 2023 esta cifra va a aumentar superando el 33% de la fuerza de trabajo. Esto implica un aumento del 
            8,3% en comparación con el 2018 y un crecimiento del 88% tomando como punto de referencia el año 2010. Es por ello que las 
            organizaciones se han enfocado en la prevención de la deserción laboral, utilizando para ello modelos basados en regresión logística, 
            redes neuronales, árboles de clasificación, bagging, random forest, gradient boosting, extreme gradient boosting, maquinas de soporte 
            vectorial, modelos ensamblados como herramientas que permitan predecir quién, cuándo y por qué un empleado pondrá fin a su carrera 
            profesional (Rey Caldeyro 2021).

            En este orden de ideas, en el presente trabajo se presenta inicialmente un análisis exploratorio de datos de una base de datos de 
            deserción de IBM. Seguidamente se entrena y se evalua modelos de aprendizaje automático; a saber, regresión logística, Random Forest, 
            máquina de soporte vectorial (SVM), K-vecino más cercano (KNN), árbol de decisión y Gaussian Naive Bayes; con el objeto de detectar 
            con precisión la deserción para ayudar a cualquier organización a mejorar las diferentes estrategias de retención de empleados cruciales.

            ## Procedencia de los datos
            
            La base de datos utiliza es propiedad de la empresa IBM. Consta de 1470 registros y 35 variables, incluida la variable de interés Rotación laboral (Attrition). 
            26 corresponden a categoricas y 9 a numericas. Para cada tipo de variables se descartó la existencia de valores nulos, es decir los 
            datos no tienen valores faltantes.

            El objetivo de la base de datos, es servir como instrumento para predecir la deserción de colaboradores a partir de características 
            personales y profesionales.

            ''', mathjax=True),
                    html.H3('Análisis de la variable de interés'),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=fig
                    )
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Markdown('''
                        A partir de la figura se observa que de los 1470 registros de la base de datos, 
                        la cantidad de colaboradores que no rotaron tiene mayor 
                        representación con un porcentaje del 83.9%; mientras que la categoría de colaboradores
                        que rotaron representa únicamente el 16.1%. 
                        Lo anterior evidencia que el conjunto de datos se encuentra desbalanceado, con 
                        menor representación de la categoría que se quiere predecir, lo que puede resultar 
                        en un problema en el rendimiento de los modelos de predicción, ya que la categoría 
                        minoritaria puede ser mayormente ignorada, lo cual limitaría la aplicación del 
                        modelo para predecir la deserción de los colaboradores de la empresa.
                        ''', mathjax=True)
                    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figgen
                    )
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figsex
                    )
                    ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figsalsex
                    )
                    ], style={'width': '49%', 'display': 'inline-block'}),
                    html.Div([
                        dcc.Markdown('''
                        El 63.3% de los colaboradores son hombres, mientras que las mujeres corresponden al 
                        36.7%. De igual forma, se evidencia que los hombres son los que mayor número de 
                        rotaciones han presentado. Respecto al ingreso mensual se encontró que para los 
                        dos géneros son similares.
                        ''', mathjax=True)
                    ], style={'width': '100%', 'display': 'inline-block', 'float': 'right'}),
                    html.Div([
                    ],style={'width': '100%', 'display': 'inline-block'}),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figage
                    )
                    ]),
                    html.Div([
                        dcc.Markdown('''
                        ### ROTACIÓN VS RANGO ETAREO
                        Los colaboradores con un rango de edad entre los 30 y 35 años es el grupo más representativo 
                        en la organización. Adicionalmente se encontró que independientemente de si se es hombre o 
                        mujer los colaboradores cuyo grupo atareo es menor a 35 años son los que tienen mayor 
                        porcentaje de rotación.
                        ''', mathjax=True)
                    ]),
                    html.Div([
                    dcc.Graph(
                    id='example-graph',
                    figure=figsal
                    )
                    ]),
                    html.Div([
                        dcc.Markdown('''
                        ### ROTACIÓN VS EXPERIENCIA LABORAL
                        Se encontró que a menor experiencia laboral de los colaboradores mayor es la tasa de rotación, 
                        siendo os colaboradores con menos de cinco años de servico son los que mas rotan. Así mismo, se 
                        evidenció que a menor años de servicio, menor es el ingreso mensual, lo que puede sugerir que uno 
                        de los motivos de las rotaciones de los colaboradores es mejorar sus ingresos, tambien se observó 
                        que a mayor experiencia laboral mayor es el nivel del trabajo.
                        ''', mathjax=True)
                    ]),
                ])  
    elif tab == 'tab-2-graph':
            return html.Div([
                html.Div([
                    html.Div([
                    html.H3('Resultados de la evaluación promedio de los modelos supervisados'),
                    generate_table(attrition_results)
                ], style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                    dcc.Graph(
                        id='example-graph1',
                        figure=figbox
                    )
                ], style={'width': '49%', 'display': 'inline-block', 'float': 'right'}),
               
                ]), 
                    html.Div([
                    ],style={'width': '100%', 'display': 'inline-block'}),

                html.Div([
                dcc.Dropdown(
                    id='model1',
                    options=[{'label': i, 'value': i} for i in names],
                    value=names[0]
                ),
                dcc.Graph(
                        id='example-graph1',
                        figure=figlogreg
                )
                ],style={'width': '49%', 'display': 'inline-block'}),
                html.Div([
                dcc.Dropdown(
                    id='model2',
                    options=[{'label': i, 'value': i} for i in names],
                    value=names[1]
                ),
                dcc.Graph(
                        id='example-graph2',
                        figure=figrf
                )
                ],style={'width': '49%', 'display': 'inline-block', 'float': 'right'})
            ])     
        
@app.callback(dash.dependencies.Output('example-graph1', 'figure'),
              [dash.dependencies.Input('model1', 'value')])
def render_content(model1):
    return actualizar_grafico(model1)

@app.callback(dash.dependencies.Output('example-graph2', 'figure'),
              [dash.dependencies.Input('model2', 'value')])
def render_content(model2):
    return actualizar_grafico(model2)

@app.callback(
    Output('display-selected-values', 'children'),
    Input('tabs-graph', 'value'),
    Input('model1', 'value'),
    Input('model2', 'value')
    )
def set_display_children(tab, model1, model2):
    if tab == 'tab-2-graph':
        return comparate_models(model1, model2)

if __name__ == '__main__':
    app.run_server(debug=True)