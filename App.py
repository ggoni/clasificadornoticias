import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
import re
import spacy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn-poster')
import pickle
import joblib
import os

import streamlit as st


import plotly.express as px
import plotly.graph_objects as go

nlp = spacy.load('es_core_news_lg')

nuevaNoticia = None


#Paleta de colores
colores_topicos = ["red", "green", "blue", "goldenrod", "magenta"]

#%config InlineBackend.figure_format = 'retina'
#sns.color_palette("crest", as_cmap=True)



clasificador = pickle.load(open('clasificador.sav', 'rb'))
vectorizador  = pickle.load(open('vectorizador.sav', 'rb'))


def limpia(word):
    word = word.lower()
    word = re.sub(r'\'', '', word)
    word = re.sub(r'\"', '', word)
    word = re.sub(r'\.', '', word)
    word = re.sub(r'\:', '', word)
    word = re.sub(r'\;', '', word)
    word = re.sub(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '',
                  word, flags=re.MULTILINE)
    word = re.sub(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', "", word)
    word = re.sub(r'ee.uu', 'eeuu', word)
    word = re.sub(r'\#\.', '', word)
    word = re.sub(r'\n', '', word)
    word = re.sub(r',', '', word)
    word = re.sub(r'\-', ' ', word)
    word = re.sub(r'\.{3}', ' ', word)
    word = re.sub(r'a{2,}', 'a', word)
    word = re.sub(r'é{2,}', 'é', word)
    word = re.sub(r'i{2,}', 'i', word)
    word = re.sub(r'ja{2,}', 'ja', word)
    word = re.sub(r'á', 'a', word)
    word = re.sub(r'é', 'e', word)
    word = re.sub(r'í', 'i', word)
    word = re.sub(r'ó', 'o', word)
    word = re.sub(r'ú', 'u', word)
    # word = re.sub('[^a-zA-Z]', ' ', word)
    word = re.sub(r'[0-9]', '', word).strip()
    
    return word


def miTokenizador(texto):
    lista = []
    for token in nlp(limpia(texto)):
        if token.pos_ in ['NOUN', 'VERB', 'PNOUN']:
            lista.append(token.lemma_)
    
    return ' '.join(lista)




def resumenNoticias(df):
    
    fig = px.bar(df,
                 x='time',
                 y='# Noticias',
                 color = 'Tópico preponderante',
                 color_discrete_sequence=colores_topicos,
                 barmode='stack')

    fig.update_layout(title="Composición histórica de noticias", height= 600, width=900)

    return fig


    
diccionario_topicos = {0:'Agenda legislativa',
                       1:'Sanitario',
                       2:'Economía',
                       3:'Vulneración de Derechos',
                       4:'Trabajo'}


def prediceNuevo(texto):
    texto = miTokenizador(texto)
    X_new = vectorizador.transform([texto])
    nmf_features_new = clasificador.transform(X_new)
    predominancia_topicos = pd.DataFrame(nmf_features_new).T

   
    predominancia_topicos.columns = ['valor']
    predominancia_topicos.valor = predominancia_topicos.valor.astype('float')
    
    predominancia_topicos['topico'] = diccionario_topicos.values()
    predominancia_topicos.sort_values(by='valor', ascending=True, inplace=True)
    
    topico_principal = str(predominancia_topicos.loc[predominancia_topicos.valor == predominancia_topicos.valor.max(),
                                                     'topico'].values[0])
    
    fig = go.Figure(go.Bar(
        x=predominancia_topicos.valor.round(4),
        y=predominancia_topicos.topico,
        orientation='h'))

    fig.update_traces(marker_color='rgb(73,113,27)', marker_line_color='rgb(45,77,14)',
                  marker_line_width=1.5, opacity=0.8)
                  
    fig.update_layout(title= f"El tópico principal es {topico_principal}")
    
    return fig


st.markdown("# Nuevas Noticias")



nuevaNoticia = st.text_area('Ingrese una nueva noticia aca:')

if len(nuevaNoticia)>0:
    st.plotly_chart(prediceNuevo(nuevaNoticia))

