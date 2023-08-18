import streamlit as st
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(layout="wide")
st.title("Creación de Reportes Sobre Leads y Efectividad de Citas")

tab1, tab2 = st.tabs(["Curva de Aprendizaje", "Monte Carlo"])

with tab1:
    st.header("Curva de Aprendizaje")

    cantidad_unidades = st.number_input("Cantidad de Unidades")
    nivel_curva = st.number_input("Nivel de la curva de aprendizaje")
    tiempo_primera_unidad = st.number_input("Tiempo para terminar primera unidad")

    tiempo_por_unidad = np.zeros(int(cantidad_unidades))
    if (cantidad_unidades > 0 and nivel_curva > 0 and tiempo_primera_unidad > 0):
        tiempo_por_unidad[0] = tiempo_primera_unidad
        for i in range(1,len(tiempo_por_unidad)):
            tiempo_por_unidad[i] = tiempo_primera_unidad * (i + 1) ** (np.log(nivel_curva) / np.log(2))

        x = np.arange(0,len(tiempo_por_unidad),1)

        fig, ax = plt.subplots()
        ax.scatter(x=x, y=tiempo_por_unidad)
        ax.set_ylabel('Tiempo para unidad')
        ax.set_xlabel('Unidad')
        ax.set_title('Curva de Aprendizaje')
        st.pyplot(fig, use_container_width=True)

        for i in range(len(tiempo_por_unidad)):
            f'''unidad {i + 1} - {tiempo_por_unidad[i]} '''

with tab2:
    st.header("Monte Carlo")
