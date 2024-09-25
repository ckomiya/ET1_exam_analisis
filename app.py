import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import openai  # Importar el cliente de OpenAI
import os  # Importar para acceder a las variables de entorno

# Configurar la clave de API de OpenAI desde una variable de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")




# Modificar la sección para construir información para las preguntas

# Modificar la sección para construir información para las preguntas
def obtener_informacion_preguntas(df_notas, df_rubrica):
    informacion_preguntas = []
    for i in range(1, 4):  # Para P1, P2, P3
        puntajes_pregunta = df_notas[f'P{i}'].dropna()  # Obtener las puntuaciones de la pregunta i
        puntos_maximos = df_rubrica.loc[df_rubrica['nro'] == i, 'puntos'].values[0]
        
        # Contar cuántos alumnos tienen cada puntuación
        conteo_puntajes = puntajes_pregunta.value_counts().sort_index()

        # Crear una representación legible con cantidad de alumnos, puntos obtenidos y puntos en la rúbrica
        representacion = [
            f"- {cantidad} alumno(s) obtuvieron {puntos_obtenidos} puntos (máximo {puntos_maximos})"
            for puntos_obtenidos, cantidad in conteo_puntajes.items()
        ]
        
        descripcion_pregunta = df_rubrica.loc[df_rubrica['nro'] == i, 'descripcion'].values[0]

        # Añadir a la lista la descripción y el conteo de puntuaciones
        informacion_preguntas.append(f"Pregunta: {i}\nDescripción: {descripcion_pregunta}\n" + "\n".join(representacion))
    
    return "\n\n".join(informacion_preguntas)






# Actualizar el prompt en la función obtener_analisis
def obtener_analisis(descripcion, media, desviacion, cantidad, nota_maxima, nota_minima, informacion_preguntas):
    # Crear el prompt para enviar al modelo GPT-4
    prompt = f"""
    Estamos analizando las notas de un examen T1, el primero de cuatro evaluaciones para un curso que se aprueba con una nota mínima de 13. Las métricas calculadas son las siguientes:
    - Media: {media:.2f}
    - Desviación estándar: {desviacion:.2f}
    - Número de alumnos: {cantidad}
    - Nota máxima: {nota_maxima:.2f}
    - Nota mínima: {nota_minima:.2f}

    A continuación, solicita lo siguiente:

    1. Explica brevemente la distribución de las notas utilizando la desviación estándar calculada para este T1.

    2. Haz un análisis de las puntuaciones que los alumnos han obtenido en este examen T1. Considerando las siguientes tres preguntas:
        {informacion_preguntas}
       El análisis debe ser tu opinión acerca del dominio del tema y desempeño que han tenido los alumnos en ese examen
    3. El curso se aprueba con una nota final mínima de 13, y esta nota final se calcula con la fórmula:
        Promedio Final = 15% (T1) + 20% (T2) + 35% (Examen Final - EF) + 30% (T3).
       Los estudiantes tienen la opción de sustituir su calificación de T1 o T2 después de realizar T2 y antes del EF y T3.

       A partir de los resultados de T1, ¿cuál sería la mejor estrategia para los alumnos con peor rendimiento en T1 para que puedan aprobar el curso?
    """

    # Hacer la llamada al modelo GPT-4
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un experto en análisis de datos educativos."},
            {"role": "user", "content": prompt},
        ],
    )
    print(prompt)
    # Obtener el texto de la respuesta
    insights = response.choices[0].message.content
    return insights


# Título de la aplicación
st.title('Análisis de Notas de Alumnos')

# Subir archivo CSV de la rúbrica de examen
archivo_rubrica = st.file_uploader("Sube el archivo CSV de la Rúbrica de Examen", type=["csv"])

# Subir archivo CSV de las notas de examen
archivo_notas = st.file_uploader("Sube el archivo CSV de las Notas de Examen", type=["csv"])

if archivo_rubrica is not None and archivo_notas is not None:
    # Leer los datos desde los archivos CSV subidos
    df_rubrica = pd.read_csv(archivo_rubrica)
    df_notas = pd.read_csv(archivo_notas)

    # Asegurarse de que solo se utiliza la columna 'nota' en el DataFrame de notas
    notas = df_notas['nota'].dropna()  # Eliminar valores nulos si existen

    # Mostrar el DataFrame de la columna 'nota'
    st.subheader('Notas de los Alumnos')

    # Crear el histograma de las notas
    fig = px.histogram(notas, nbins=20, title='Distribución de Notas de los Alumnos')

    # Asegurar que se muestren todos los valores en el eje X (cada número de nota)
    fig.update_xaxes(dtick=1)

    # Asegurar que el eje Y solo muestre valores enteros y que no se repitan
    fig.update_yaxes(tickformat="d", tickmode='linear', dtick=1)

    # Calcular estadísticas
    desviacion_estandar = notas.std()
    media = notas.mean()
    cantidad_alumnos = notas.count()
    nota_maxima = notas.max()  # Nota máxima
    nota_minima = notas.min()  # Nota mínima

    # Calcular los rangos y el porcentaje de alumnos en cada rango
    rango_bajo = media - desviacion_estandar
    rango_alto = media + desviacion_estandar

    # Contar cuántos alumnos están en cada rango
    total_bajo = notas[notas < rango_bajo].count()
    total_medio = notas[(notas >= rango_bajo) & (notas <= rango_alto)].count()
    total_alto = notas[notas > rango_alto].count()

    # Calcular los porcentajes
    porcentaje_bajo = (total_bajo / cantidad_alumnos) * 100
    porcentaje_medio = (total_medio / cantidad_alumnos) * 100
    porcentaje_alto = (total_alto / cantidad_alumnos) * 100

    # Datos para el gráfico de porcentajes
    etiquetas = [f'Por debajo de {rango_bajo:.2f}', f'Entre {rango_bajo:.2f} y {rango_alto:.2f}', f'Por encima de {rango_alto:.2f}']
    porcentajes = [porcentaje_bajo, porcentaje_medio, porcentaje_alto]

    # Crear un gráfico de barras
    bar_fig = go.Figure(data=[
        go.Bar(x=etiquetas, y=porcentajes, marker_color='royalblue')
    ])

    # Agregar título y etiquetas al gráfico
    bar_fig.update_layout(title='Porcentaje de Alumnos por Rango',
                          xaxis_title='Rango de Notas',
                          yaxis_title='Porcentaje (%)')

    # Análisis de preguntas
    st.subheader('Análisis de Preguntas')

    # Crear columnas para los gráficos de las preguntas
    col1, col2, col3, col4 = st.columns(4)

    for i in range(1, 4):  # Para P1, P2, P3
        puntajes_pregunta = df_notas[f'P{i}'].dropna()  # Obtener las puntuaciones de la pregunta i
        puntos_maximos = df_rubrica.loc[df_rubrica['nro'] == i, 'puntos'].values[0]
        descripcion_pregunta = df_rubrica.loc[df_rubrica['nro'] == i, 'descripcion'].values[0]

        # Asegúrate de convertir puntos_maximos a int
        puntos_maximos = int(puntos_maximos)

        # Crear el histograma para la pregunta
        fig_pregunta = px.histogram(puntajes_pregunta, nbins=puntos_maximos, title=f'Pregunta {i}',
                             labels={'value': f'Pregunta {i}'})

        # Mostrar el gráfico de distribución de puntuaciones en la columna correspondiente
        if i == 1:
            col1.plotly_chart(fig_pregunta, use_container_width=True)
            col1.write(f"Descripción: {descripcion_pregunta}")
        elif i == 2:
            col2.plotly_chart(fig_pregunta, use_container_width=True)
            col2.write(f"Descripción: {descripcion_pregunta}")
        elif i == 3:
            col3.plotly_chart(fig_pregunta, use_container_width=True)
            col3.write(f"Descripción: {descripcion_pregunta}")

    # Gráfico en la cuarta columna utilizando los valores de la columna 'PX'
    puntajes_PX = df_notas['PX'].dropna()  # Obtener las puntuaciones de la columna 'PX'
    puntos_maximos_PX = df_rubrica['puntos'].max()  # Asumir que 'puntos' de la rúbrica se aplica a 'PX'

    # Asegúrate de convertir puntos_maximos_PX a int
    puntos_maximos_PX = int(puntos_maximos_PX)

    # Crear el histograma para la columna 'PX'
    fig_PX = px.histogram(puntajes_PX, nbins=puntos_maximos_PX, title='Criterios Proyecto',
                           labels={'value': 'Puntuaciones de PX'})

    # Mostrar el gráfico de distribución de puntuaciones de PX en la cuarta columna
    col4.plotly_chart(fig_PX, use_container_width=True)

    # Mostrar el gráfico de distribución de notas
    st.plotly_chart(fig, use_container_width=True)

    # Mostrar los datos y estadísticas
    st.subheader('Estadísticas de Notas')
    st.write(f"Desviación Estándar: {desviacion_estandar:.2f}")
    st.write(f"Media: {media:.2f}")
    st.write(f"Número de Alumnos: {cantidad_alumnos}")
    st.write(f"Nota Máxima: {nota_maxima:.2f}")  # Nota máxima
    st.write(f"Nota Mínima: {nota_minima:.2f}")  # Nota mínima

    # Mostrar el gráfico de porcentajes
    st.plotly_chart(bar_fig, use_container_width=True)
    
    # Descripciones de las preguntas P1, P2 y P3
    descripcion_preguntas = [
        df_rubrica.loc[df_rubrica['nro'] == 1, 'descripcion'].values[0],
        df_rubrica.loc[df_rubrica['nro'] == 2, 'descripcion'].values[0],
        df_rubrica.loc[df_rubrica['nro'] == 3, 'descripcion'].values[0]
    ]

    # Botón para obtener el análisis y descargarlo
    if st.button("Generar Análisis y Descargar"):
        # Llamar a la función para obtener la información de las preguntas antes de crear el análisis
        informacion_preguntas = obtener_informacion_preguntas(df_notas, df_rubrica)
        analisis_resultado = obtener_analisis(
            descripcion=None,
            media=media,
            desviacion=desviacion_estandar,
            cantidad=cantidad_alumnos,
            nota_maxima=nota_maxima,
            nota_minima=nota_minima,
            informacion_preguntas=informacion_preguntas
        )

        # Preparar el archivo de texto para descargar
        filename = "analisis_T1.txt"
        with open(filename, 'w') as f:
            f.write(analisis_resultado)

        # Botón para descargar el archivo de análisis
        with open(filename, 'r') as f:
            st.download_button(
                label="Descargar Análisis como TXT",
                data=f,
                file_name=filename,
                mime="text/plain"
            )

