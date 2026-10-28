import streamlit as st
import numpy as np

def calc_gin(x,w,metoda):
    xw=x*w
    if metoda=="sum":
        return np.sum(xw)
    elif metoda=="prod":
        return np.prod(xw)
    elif metoda=="max":
        return np.max(xw)
    elif metoda=="min":
        return np.min(xw)

def act_lin(t, z, g): 
    return g*(z-t)
def act_sigmoidala(z,t,g):
    return 1/(1+np.exp(-g*(z-t)))
def act_signum(z,t,g):
    return np.sign(g*(z-t))
def act_tanh(z,t,g):
    return np.tanh(g*(z-t))
def act_treapta(z,t):
    return np.where((z-t)>=0, 1, 0)

def activare(z,t,g, metoda):
    if metoda=="Liniara":
        return act_lin(t,z,g)
    elif metoda=="Sigmoidala":
        return act_sigmoidala(z,t,g)
    elif metoda=="Signum":
        return act_signum(z,t,g)
    elif metoda=="Tanh":
        return act_tanh(z,t,g)
    elif metoda=="Treapta":
        return act_treapta(z,t)

def main():
    st.set_page_config(layout="wide")
    st.title("Retea Neuronala Artificiala")
    st.sidebar.header("1. Arhitectura Retelei")
    n_intrari=st.sidebar.number_input("Neuroni pe stratul de intrare (n):", min_value=1, value=2)
    n_straturi_ascunse=st.sidebar.number_input("Numarul de straturi ascunse:", min_value=1,max_value=3, value=1)

    neuroni_ascunsi=[]
    for i in range(n_straturi_ascunse):
        nr=st.sidebar.number_input(f"Numarul de neuroni pe stratul ascuns {i+1}:", min_value=1, value=2)
        neuroni_ascunsi.append(nr)

    n_iesiri=st.sidebar.number_input("Neuroni pe stratul de iesire:", min_value=1, value=1)
    st.write("Structura configurata:")
    st.info(f"Intrari: {n_intrari}, Straturi ascunse: {neuroni_ascunsi}, Iesiri: {n_iesiri}")

    st.divider()

    nr_total_straturi=1+n_straturi_ascunse+1
    coloane=st.columns(nr_total_straturi)

    x_curent=[]

    with coloane[0]:
        st.header("Stratul de intrare")
        st.caption(f"{n_intrari} neuroni")

        for i in range(n_intrari):
            x_curent.append(st.number_input(f"x{i+1}:", value=0.0, key=f"x{i}"))

        x_curent=np.array(x_curent)
        st.write("Iesirile stratului:")
        st.code(np.round(x_curent,3))

        straturi_retea= neuroni_ascunsi + [n_iesiri]
        nume_straturi=[f"Ascuns {i+1}" for i in range(n_straturi_ascunse)] + ["Iesire"]

        for idx_strat, nr_neuroni in enumerate(straturi_retea):
            with coloane[idx_strat+1]:
                st.header(nume_straturi[idx_strat])
                st.caption(f"{nr_neuroni} neuroni")

                st.markdown("Setari strat")
                metoda_gin=st.selectbox("Metoda de calcul GIN:", ["sum", "prod", "max", "min"], key=f"metoda_gin_{idx_strat}")
                act=st.selectbox("Functia de activare:", ["Liniara", "Sigmoidala", "Signum", "Tanh", "Treapta"], key=f"act_{idx_strat}")
                t=st.number_input("Pragul (t):", value=0.0, key=f"t_{idx_strat}")

                g=1.0
                if act in ["Liniara", "Sigmoidala", "Tanh"]:
                    g=st.number_input("Factorul de scalare (g):", value=1.0, key=f"g_{idx_strat}")

                tip_iesire= st.radio("Tipul iesirii:", ["Real", "Binar"], key=f"tip_iesire_{idx_strat}")

                st.divider()

                iesiri_strat=[]
                for j in range(nr_neuroni):
                    st.markdown(f"Neuron {j+1}")
                    w_neuron=[]

                    for k in range(len(x_curent)):
                        tarie= st.number_input(f"w de la intrarea {k+1}:", value=0.0, key=f"w_{idx_strat}_{j}_{k}")
                        w_neuron.append(tarie)

                    w_neuron=np.array(w_neuron)

                    z=calc_gin(x_curent,w_neuron,metoda_gin)
                    y_calc=activare(z,t,g,act)

                    if tip_iesire=="Binar":
                        if act in ["Sigmoidala", "Treapta"]:
                            y_final= 1 if y_calc>=0.5 else 0
                        else:
                            y_final= 1 if y_calc>= 0 else -1
                    else:
                        y_final = y_calc
                    iesiri_strat.append(y_final)

                    st.success(f"Valoarea iesirii (y): {np.round(y_final, 3)}")

                x_curent=np.array(iesiri_strat)

if __name__=="__main__":
    main()
        
        