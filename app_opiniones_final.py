
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from dash import Dash, dcc, html, Input, Output, State
import plotly.express as px
from transformers import pipeline
import base64
import io

# Modelos
clasificador = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
resumen_modelo = pipeline("summarization", model="facebook/bart-large-cnn")

# App
app = Dash(__name__)
app.title = "Análisis de Opiniones de Clientes"

# Layout
app.layout = html.Div([
    html.H1("🗣️ Análisis de Opiniones de Clientes", style={'textAlign': 'center'}),

    dcc.Upload(
        id='upload-data',
        children=html.Div(['Arrastra o selecciona un archivo CSV con una columna llamada "opiniones"']),
        style={
            'width': '100%', 'height': '60px', 'lineHeight': '60px',
            'borderWidth': '1px', 'borderStyle': 'dashed',
            'borderRadius': '5px', 'textAlign': 'center', 'margin': '10px'
        },
        multiple=False
    ),

    html.Div(id='output-data-upload'),

    html.H2("🔍 Analizar nuevo comentario"),
    dcc.Textarea(id='nuevo-comentario', style={'width': '100%'}, rows=4),
    html.Button("Analizar", id='boton-analizar', n_clicks=0),
    html.Div(id='salida-comentario')
])

# === Funciones auxiliares ===

def limpiar_texto(texto):
    palabras = texto.lower().split()
    return [p for p in palabras if p.isalpha() and p not in stopwords.words("spanish")]

def generar_nube(opiniones):
    texto = " ".join(opiniones)
    palabras = limpiar_texto(texto)
    nube = WordCloud(width=800, height=400, background_color='white').generate(" ".join(palabras))
    img = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(nube, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(img, format='png')
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def clasificar_sentimientos(opiniones):
    etiquetas = []
    for texto in opiniones:
        estrellas = int(clasificador(texto)[0]['label'][0])
        if estrellas >= 4:
            etiquetas.append("Positivo")
        elif estrellas == 3:
            etiquetas.append("Neutro")
        else:
            etiquetas.append("Negativo")
    return etiquetas

def analizar_comentario(texto):
    sentimiento = clasificador(texto)[0]['label']
    resumen = resumen_modelo(texto[:1024])[0]['summary_text']
    return sentimiento, resumen

# === Callbacks ===

@app.callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def procesar_csv(contents, filename):
    if contents is None:
        return
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

    if "opiniones" not in df.columns:
        return html.Div("❌ El archivo debe tener una columna llamada 'opiniones'.")

    opiniones = df["opiniones"].dropna().astype(str).tolist()
    if len(opiniones) != 20:
        return html.Div("⚠️ El archivo debe contener exactamente 20 opiniones.")

    df["Sentimiento"] = clasificar_sentimientos(opiniones)

    # Gráfico de barras de palabras
    palabras = limpiar_texto(" ".join(opiniones))
    conteo = Counter(palabras).most_common(10)
    palabras_top, frecs = zip(*conteo)
    fig_barras = px.bar(x=palabras_top, y=frecs, labels={'x': 'Palabra', 'y': 'Frecuencia'}, title="Top 10 palabras")

    # Pie chart de sentimientos
    fig_pie = px.pie(df, names="Sentimiento", title="Distribución de Sentimientos")

    # Nube de palabras
    imagen_base64 = generar_nube(opiniones)

    return html.Div([
        html.H3("📊 Top 10 Palabras"),
        dcc.Graph(figure=fig_barras),

        html.H3("☁️ Nube de Palabras"),
        html.Img(src="data:image/png;base64,{}".format(imagen_base64), style={'width': '100%'}),

        html.H3("📈 Opiniones clasificadas"),
        dcc.Graph(figure=fig_pie),

        html.H3("🧾 Opiniones con Sentimiento"),
        html.Table([
            html.Tr([html.Th("Opinión"), html.Th("Sentimiento")])
        ] + [html.Tr([html.Td(o), html.Td(s)]) for o, s in zip(df["opiniones"], df["Sentimiento"])])
    ])

@app.callback(
    Output('salida-comentario', 'children'),
    Input('boton-analizar', 'n_clicks'),
    State('nuevo-comentario', 'value')
)
def analizar_nuevo(n_clicks, texto):
    if n_clicks > 0 and texto:
        sentimiento, resumen = analizar_comentario(texto)
        return html.Div([
            html.P(f"🟢 Sentimiento estimado: {sentimiento}"),
            html.P("📝 Resumen generado:"),
            html.Blockquote(resumen)
        ])

# === Ejecutar la app ===
if __name__ == '__main__':
    app.run(debug=True)
