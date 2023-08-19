import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy_financial as npf
import scipy.stats as stats


st.set_page_config(layout="wide")
st.title("Proyecto de clase Parte 1")
st.subheader("Jonathan Corado - 18001211  Eduardo Calderón - 18002632")

tab1, tab2 = st.tabs(["Curva de Aprendizaje", "Monte Carlo"])

with tab1:
    st.header("Curva de Aprendizaje")

    cantidad_unidades = st.number_input("Cantidad de Unidades")
    nivel_curva = st.number_input("Tasa de mejora")
    tiempo_primera_unidad = st.number_input("Tiempo para terminar primera unidad")
    tipo_curva = st.radio("Tipo de valores", 
                          ("Unidades", "Acumulativos"))
    
    if (cantidad_unidades > 0 and nivel_curva > 0 and tiempo_primera_unidad > 0):
        tiempo_por_unidad = np.zeros(int(cantidad_unidades))
        tiempo_por_unidad[0] = tiempo_primera_unidad
        for i in range(1,len(tiempo_por_unidad)):
            if tipo_curva == 'Unidades':
                tiempo_por_unidad[i] = round(tiempo_primera_unidad * (i + 1) ** (np.log(nivel_curva) / np.log(2)), 2)
            else:
                tiempo_por_unidad[i] = round(tiempo_primera_unidad * (i + 1) ** (np.log(nivel_curva) / np.log(2)) + tiempo_por_unidad[i - 1], 2)

        x = np.arange(0,len(tiempo_por_unidad),1)

        tabla, grafico = st.tabs(["Tabla", "Gráficas"])

        unidades = np.arange(len(tiempo_por_unidad)) + 1
        arrFinal = np.array([unidades, tiempo_por_unidad]).transpose()
        df = pd.DataFrame(
            arrFinal,
            columns=('Unidades', 'Horas')
            )

        with grafico:
            # fig, ax = plt.subplots()
            # ax.scatter(x=x, y=tiempo_por_unidad)
            # ax.set_ylabel('Tiempo para unidad')
            # ax.set_xlabel('Unidad')
            # ax.set_title('Curva de Aprendizaje')
            # st.pyplot(fig, use_container_width=True)
            charts_df = df
            charts_df.index = np.arange(1, len(charts_df) + 1)
            st.line_chart(charts_df, x="Unidades", y="Horas")
            st.bar_chart(charts_df, x="Unidades", y="Horas")
        
        with tabla:
            N = 5
            last_page = len(unidades) // N
            if 'page_number' not in st.session_state:
                st.session_state['page_number'] = 0

            
            
            #st.dataframe(df.set_index(df.columns[0]), height=(len(unidades) + 1) * 35 + 3)
            prev, _, next = st.columns([1, 10, 1])
            st.write(st.session_state.page_number)
            if next.button("Next"):
                if st.session_state.page_number + 1 > last_page:
                     st.session_state.page_number = 0
                else:
                    st.session_state.page_number += 1

            if prev.button("Previous"):
                if st.session_state.page_number - 1 < 0:
                    st.session_state.page_number = last_page
                else:
                    st.session_state.page_number -= 1

            # Get start and end indices of the next page of the dataframe
            start_idx = st.session_state.page_number * N 
            end_idx = (1 + st.session_state.page_number) * N

            # Index into the sub dataframe
            sub_df = df.iloc[start_idx:end_idx]
            st.write(sub_df)

with tab2:
    st.header("Monte Carlo")

    tasa = st.number_input("Tasa")
    iteraciones = st.number_input("Iteraciones")

    df = pd.DataFrame(np.zeros(3).reshape(1,3), columns=('anios', 'Flujos', 'STDEV'))
    table = st.data_editor(df, num_rows="dynamic")

    boton = st.button("Simular")
    if boton:
        medias = table['Flujos'][1:len(table['Flujos'])]

        stedv = table['STDEV'][1:len(table['STDEV'])]

        vpn_fijo = npf.npv(int(tasa) / 100, np.array(table['Flujos']))

        vpns =[]
        flujo_actual = []
        capital = 0
        for j in range(int(iteraciones)):
            flujo_actual = []
            flujo_actual.append(np.random.normal(table['Flujos'][0]))
            for i in range(1, len(medias) + 1):
                flujo_actual.append(np.random.normal(medias[i], stedv[i]))

            #vpn aleatorio
            vpns.append(npf.npv(tasa / 100, flujo_actual))

        
        uno, dos = st.columns([6,6])
        with  uno :
            flujosDf = pd.DataFrame(
                flujo_actual, columns=['Flujos'])

            st.write(flujosDf)

            vpnsDf = pd.DataFrame( 
                    np.array([[vpn_fijo],[vpns[-1]]]).transpose()
                ,columns=('VPN Fijo', 'VPN Aleatorio'))
            
            st.write(vpnsDf)

        with dos:
            mu = 0
            variance = np.var(vpns)
            sigma = np.sqrt(variance)
            x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
            fig, ax = plt.subplots()
            ax.plot(x, stats.norm.pdf(x, mu, sigma))
            st.pyplot(fig)

            valoresExtras = pd.DataFrame(
                np.array([[np.max(vpns)],[np.min(vpns)],[np.mean(vpns)]]).transpose(),
                columns=('Maximo', 'Minimo', 'Mean')
            )

            st.write(valoresExtras)

    #st.write(flujo_actual)    


    #st.write(table['anios'][0])