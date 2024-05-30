import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

def obtener_datos(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)
            ticker_data.reset_index(inplace=True)
            ticker_data['Date'] = pd.to_datetime(ticker_data['Date']).dt.date
            ticker_data.set_index('Date', inplace=True)
            data[ticker] = ticker_data.sort_index(ascending=False)  # Ordenar los datos por fecha descendente
        except Exception as e:
            st.error(f"No se pudieron obtener datos para {ticker}: {str(e)}")
    return data

def calcular_rendimientos_esperados_volatilidad(datos, pesos):
    rendimientos = np.log(datos / datos.shift(1))
    rendimientos_esperados = np.sum(rendimientos.mean() * pesos) * 252
    covarianza = rendimientos.cov() * 252
    volatilidad = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
    return rendimientos_esperados, volatilidad, covarianza

def simular_montecarlo(rendimientos_logaritmicos, num_simulaciones, num_anios):
    resultados = np.zeros((3+len(rendimientos_logaritmicos.columns), num_simulaciones))
    rendimientos = rendimientos_logaritmicos.values
    
    for i in range(num_simulaciones):
        # Generar pesos aleatorios
        pesos = np.random.random(len(rendimientos_logaritmicos.columns))
        pesos /= np.sum(pesos)

        # Calcular rendimiento y volatilidad del portafolio
        rendimiento_portafolio = np.sum(rendimientos.mean() * pesos) * 252
        covarianza = np.cov(rendimientos.T) * 252
        volatilidad_portafolio = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

        # Sharpe Ratio
        sharpe_ratio = rendimiento_portafolio / volatilidad_portafolio

        # Almacenar resultados
        resultados[0,i] = rendimiento_portafolio
        resultados[1,i] = volatilidad_portafolio
        resultados[2,i] = sharpe_ratio

        for j in range(len(pesos)):
            resultados[j+3,i] = pesos[j]

    return resultados

def verificar_covarianza_baja(covarianza):
    umbral = 0.05  # Umbral para considerar una covarianza baja
    covarianza_promedio = covarianza.mean().mean()
    if covarianza_promedio < umbral:
        return True
    else:
        return False

def pagina_principal():
    st.title("GENERA TU CARTERA DE INVERSIÓN IDEAL")
    st.write("¡Bienvenido a nuestra plataforma!")
    st.write("Aquí usamos el modelo de Markowitz para ayudarte a construir una cartera de inversiones que maximice tus rendimientos sin arriesgar demasiado.")
    st.write("El modelo de Markowitz diversifica tu cartera asignando pesos óptimos a diferentes activos para encontrar el equilibrio entre riesgo y rendimiento.")

    aversion_riesgo = st.radio(
        "Selecciona tu Nivel de Aversión al Riesgo:",
        ('Conservador', 'Moderado', 'Acepto riesgos'))

    if aversion_riesgo == 'Conservador':
        st.write("Prefieres opciones de inversión seguras y estables, priorizando la preservación del capital sobre los altos rendimientos.")
    elif aversion_riesgo == 'Moderado':
        st.write("Estás dispuesto a asumir un riesgo moderado en tus inversiones en busca de un equilibrio entre seguridad y rendimiento.")
    elif aversion_riesgo == 'Acepto riesgos':
        st.write("Estás dispuesto a asumir un nivel considerable de riesgo en tus inversiones en busca de mayores rendimientos.")

    st.markdown("---")

    st.header("COLOCA LA INVERSIÓN POR REALIZAR")

    inversion_deseada = st.text_input("Cantidad de Inversión")

    st.markdown("---")

    st.title("ELIGE LAS ACCIONES PARA ARMAR TU PORTAFOLIO")
    st.write("Selecciona las acciones de las cuales deseas obtener los rendimientos logarítmicos.")

    nuevo_activo = st.text_input("Escribe el ticker de la acción que deseas añadir (por ejemplo, AAPL)")

    if st.button("Añadir"):
        if nuevo_activo in st.session_state.activos_seleccionados:
            st.error("¡Este activo ya ha sido seleccionado!")
        else:
            st.session_state.activos_seleccionados.append(nuevo_activo)

    if st.button("Reiniciar Portafolio"):
        st.session_state.activos_seleccionados = []

    st.header("COMPOSICIÓN DEL PORTAFOLIO:")
    with st.expander("Información de las empresas", expanded=True):
        for activo in st.session_state.activos_seleccionados:
            mostrar_info_activo(activo)

    if st.session_state.activos_seleccionados:
        if st.button("Descargar Datos"):
            fecha_hoy = datetime.datetime.now().date()
            fecha_inicio = fecha_hoy - datetime.timedelta(days=2*365)  

            datos = obtener_datos(st.session_state.activos_seleccionados, fecha_inicio, fecha_hoy)
            if datos:
                rendimientos_logaritmicos = pd.DataFrame()
                fig, ax = plt.subplots(figsize=(10, 6))
                for ticker, df in datos.items():
                    st.subheader(f"Datos para {ticker}")
                    st.write(df)
                    rendimientos_logaritmicos[ticker] = df['Close'].pct_change().dropna()

                    # Gráfico de datos históricos
                    ax.plot(df.index, df['Close'], label=ticker)

                ax.set_xlabel('Fecha')
                ax.set_ylabel('Precio de Cierre')
                ax.set_title('Datos Históricos de Acciones (Año en curso)')
                ax.legend()
                plt.tight_layout()  # Ajuste de diseño para habilitar el zoom
                st.pyplot(fig)

                # Limitar los datos históricos al año en curso
                fecha_inicio_anio = datetime.datetime(datetime.datetime.now().year, 1, 1).date()
                datos_anio_actual = obtener_datos(st.session_state.activos_seleccionados, fecha_inicio_anio, fecha_hoy)
                
                rendimientos_logaritmicos_anio_actual = pd.DataFrame()
                for ticker, df in datos_anio_actual.items():
                    rendimientos_logaritmicos_anio_actual[ticker] = df['Close'].pct_change().dropna()

                # Cálculo de los rendimientos logarítmicos
                st.title("RENDIMIENTOS LOGARITMICOS DE LOS ACTIVOS (Año en curso)")
                st.write(rendimientos_logaritmicos_anio_actual)

                # Gráfico de rendimientos logarítmicos
                st.line_chart(rendimientos_logaritmicos_anio_actual)

                pesos = np.ones(len(st.session_state.activos_seleccionados)) / len(st.session_state.activos_seleccionados)
                rendimientos_esperados, volatilidad, covarianza = calcular_rendimientos_esperados_volatilidad(rendimientos_logaritmicos_anio_actual, pesos)
                st.subheader("Covarianza Anualizada de los Rendimientos (Año en curso)")
                st.write(covarianza)
                
                if verificar_covarianza_baja(covarianza):
                    st.write("La covarianza entre los activos seleccionados es baja, lo cual es óptimo según el modelo de Markowitz.")
                else:
                    st.warning("La covarianza entre los activos seleccionados no es baja. Considera ajustar tu selección de activos para mejorar la diversificación.")

                    # Calcular el coeficiente de correlación
                    correlacion = rendimientos_logaritmicos_anio_actual.corr()

                    # Mostrar el coeficiente de correlación
                    st.subheader("Coeficiente de Correlación entre los Rendimientos (Año en curso)")
                    st.write(correlacion)

                    # Explicación del coeficiente de correlación
                    st.warning("El coeficiente de correlación mide la fuerza y la dirección de la relación lineal entre los rendimientos de los activos.")
                    st.write("Cuando el coeficiente es 1, indica una correlación positiva perfecta, lo que significa que los activos tienden a moverse en la misma dirección.")
                    st.write("Cuando el coeficiente es -1, indica una correlación negativa perfecta, lo que significa que los activos tienden a moverse en direcciones opuestas.")
                    st.write("Cuando el coeficiente es 0, indica una correlación nula, lo que significa que no hay relación lineal entre los movimientos de los activos.")
                
                # Simulación de Monte Carlo
                st.title("SIMULACIÓN MONTECARLO")
                num_simulaciones = 10000
                num_anios = 5
                resultados_simulacion = simular_montecarlo(rendimientos_logaritmicos_anio_actual, num_simulaciones, num_anios)
                
                mejor_sharpe_index = np.argmax(resultados_simulacion[2])
                mejor_sharpe_rendimiento = resultados_simulacion[0, mejor_sharpe_index]
                mejor_sharpe_volatilidad = resultados_simulacion[1, mejor_sharpe_index]
                mejor_sharpe_pesos = resultados_simulacion[3:, mejor_sharpe_index]

                st.subheader("PORTAFOLIO MÁS EFICIENTE (según mayor sharpe)")
                st.write(f"Rendimiento Esperado: **{mejor_sharpe_rendimiento:.2%}**")
                st.write(f"Volatilidad Esperada: **{mejor_sharpe_volatilidad:.2%}**")
                st.write(f"Sharpe Ratio: **{mejor_sharpe_rendimiento / mejor_sharpe_volatilidad:.4f}**")

                # Calcular la inversión por activo en el portafolio eficiente
                if inversion_deseada:
                    inversiones_por_activo = [float(inversion_deseada.replace(",", "")) * peso for peso in mejor_sharpe_pesos]
                    st.subheader("INVERSIÓN EN CADA ACTIVO")
                    for i, activo in enumerate(st.session_state.activos_seleccionados):
                        st.write(f"Inversión en {activo}: **${inversiones_por_activo[i]:,.2f}**")
                else:
                    st.warning("Por favor ingresa la cantidad de inversión deseada.")

                # Desglose de inversión por activo en la barra lateral
                st.sidebar.subheader("Desglose de Inversión por Activo:")
                for i, activo in enumerate(st.session_state.activos_seleccionados):
                    st.sidebar.write(f"Inversión en {activo}: **${inversiones_por_activo[i]:,.2f}**")

                # Gráfico de pastel con los pesos correspondientes
                plt.figure(figsize=(8, 8))
                plt.pie(mejor_sharpe_pesos, labels=st.session_state.activos_seleccionados, autopct='%1.1f%%')
                plt.title("Ponderación de Activos en el Portafolio Eficiente")
                st.pyplot(plt)

                # Aviso sobre la volatilidad del portafolio eficiente
                st.subheader("Evaluación del Nivel de Volatilidad del Portafolio:")
                if aversion_riesgo == 'Conservador' and mejor_sharpe_volatilidad > 0.1:
                    st.warning("La volatilidad del portafolio eficiente podría ser demasiado alta para un inversor conservador. Se recomienda revisar la asignación de activos.")
                elif aversion_riesgo == 'Moderado' and mejor_sharpe_volatilidad > 0.2:
                    st.warning("La volatilidad del portafolio eficiente podría ser alta para un inversor moderado. Se recomienda revisar la asignación de activos.")
                elif aversion_riesgo == 'Acepto riesgos' and mejor_sharpe_volatilidad > 0.3:
                    st.warning("La volatilidad del portafolio eficiente podría ser alta incluso para un inversor que acepta riesgos. Se recomienda revisar la asignación de activos.")

            else:
                st.error("No se pudieron obtener datos para los activos seleccionados.")
    else:
        st.warning("No hay activos seleccionados. Por favor, selecciona al menos uno.")

    st.sidebar.subheader("**Perfil del Inversionista**")
    st.sidebar.write("**Nivel de Aversión al Riesgo:**", aversion_riesgo)
    st.sidebar.write("**Cantidad de Inversión Deseada:**", "{:,}".format(int(inversion_deseada.replace(",", ""))) if inversion_deseada else "")
    st.sidebar.write("**Activos Seleccionados:**")
    for activo in st.session_state.activos_seleccionados:
        st.sidebar.write("- ", activo)

def mostrar_info_activo(ticker):
    st.subheader(f"Información para {ticker}")
    
    nombre_empresa, resumen = obtener_resumen_ejecutivo(ticker)
    st.write("- **Nombre de la Empresa:**", nombre_empresa, f"({ticker})")
    st.write("  **Resumen Ejecutivo:**", resumen[:500])
    
    info = yf.Ticker(ticker).info
    st.subheader("Datos Fundamentales")
    fundamentales = {
        "Market Cap": info.get('marketCap', 'N/A'),
        "Beta": info.get('beta', 'N/A'),
        "Precio de Cierre": info.get('previousClose', 'N/A')
    }
    df = pd.DataFrame.from_dict(fundamentales, orient='index', columns=['Valor'])
    st.write(df)

def obtener_resumen_ejecutivo(ticker):
    info = yf.Ticker(ticker).info
    nombre_empresa = info.get('longName', '')
    resumen = info.get('longBusinessSummary', '')
    return nombre_empresa, resumen

def main():
    if 'activos_seleccionados' not in st.session_state:
        st.session_state.activos_seleccionados = []

    pagina_principal()

if __name__ == "__main__":
    main()
