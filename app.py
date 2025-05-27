import nltk
nltk.data.path.append("nltk_data")  # Usar carpeta local para evitar descarga en Render

from nltk.corpus import stopwords
from collections import Counter
import pandas as pd
import dash
from dash import html, dcc, Input, Output
import plotly.express as px
from wordcloud import WordCloud
import base64
import io

# Leer opiniones
df = pd.read_csv("opiniones_con_neutro.csv")
stop_words = set(stopwords.words("spanish"))

# Tokenización y filtrado
all_words = " ".join(df["Opinion"]).lower().split()
filtered_words = [word for word in all_words if word.isalpha() and word not in stop_words]

# Conteo de palabras
word_counts = Counter(filtered_words)
top_words = word_counts.most_common(10)

# Generar nube de palabras como imagen base64
wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(filtered_words))
buf = io.BytesIO()
wc.to_image().save(buf, format="PNG")
encoded_wc = base64.b64encode(buf.getvalue()).decode()

# Clasificación de sentimientos (ya debe estar en tu CSV)
sentiment_counts = df["Sentimiento"].value_counts().reset_index()
sentiment_counts.columns = ["Sentimiento", "Cantidad"]

# App Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Análisis de Opiniones", style={"textAlign": "center"}),

    html.Div([
        html.H3("📊 Top 10 Palabras Más Frecuentes"),
        dcc.Graph(
            figure=px.bar(
                x=[w[0] for w in top_words],
                y=[w[1] for w in top_words],
                labels={"x": "Palabra", "y": "Frecuencia"},
                title="Top 10 palabras más comunes"
            )
        ),
    ]),

    html.Div([
        html.H3("☁️ Nube de Palabras"),
        html.Img(src="data:image/png;base64,{}".format(encoded_wc), style={"width": "70%", "margin": "auto", "display": "block"}),
    ]),

    html.Div([
        html.H3("📈 Sentimientos"),
        dcc.Graph(
            figure=px.pie(
                sentiment_counts,
                values="Cantidad",
                names="Sentimiento",
                title="Distribución de sentimientos"
            )
        ),
    ]),

    html.Div([
        html.H3("🔍 Tabla de Opiniones"),
        dcc.Dropdown(
            id="filtro_sentimiento",
            options=[{"label": s, "value": s} for s in df["Sentimiento"].unique()] + [{"label": "Todos", "value": "Todos"}],
            value="Todos",
            placeholder="Filtrar por sentimiento"
        ),
        html.Br(),
        html.Div(id="tabla_opiniones")
    ])
])


@app.callback(
    Output("tabla_opiniones", "children"),
    Input("filtro_sentimiento", "value")
)
def actualizar_tabla(filtro):
    if filtro == "Todos":
        filtrado = df
    else:
        filtrado = df[df["Sentimiento"] == filtro]

    return html.Table([
        html.Tr([html.Th(col) for col in filtrado.columns])
    ] + [
        html.Tr([html.Td(filtrado.iloc[i][col]) for col in filtrado.columns])
        for i in range(min(len(filtrado), 10))  # Mostrar máx 10
    ])


# === Ejecutar la app ===
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=False)
    
if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8080, debug=False)

