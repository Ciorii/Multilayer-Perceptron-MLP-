import numpy as np
import streamlit as st
import pandas as pd


def calc_gin(x, w, metoda):
    xw = x * w
    if metoda == "sum":
        return np.sum(xw)
    elif metoda == "prod":
        return np.prod(xw)
    elif metoda == "max":
        return np.max(xw)
    elif metoda == "min":
        return np.min(xw)


def act_lin(t, z, g):
    return g * (z - t)


def act_sigmoidala(z, t, g):
    return 1 / (1 + np.exp(-g * (z - t)))


def act_signum(z, t):
    return np.sign(z - t)


def act_tanh(z, t, g):
    return np.tanh(g * (z - t))


def act_treapta(z, t):
    return np.where((z - t) >= 0, 1, 0)


def main():
    st.title("Neuron Artificial")

    n = st.number_input("Numarul de intrari (n):", min_value=1, value=2, step=1)

    x = []
    w = []

    col1, col2 = st.columns(2)

    for i in range(n):
        with col1:
            x.append(st.number_input(f"x{i + 1}:", value=0.0, key=f"x{i}"))
        with col2:
            w.append(st.number_input(f"w{i + 1}:", value=0.0, key=f"w{i}"))

    w = np.array(w)
    x = np.array(x)

    metoda_gin = st.selectbox("Metoda de calcul:", ["sum", "prod", "max", "min"])
    z = calc_gin(x, w, metoda_gin)
    st.info(f"Valoare GIN: {z}")

    act = st.selectbox(
        "Functia de activare:",
        ["Liniara", "Sigmoidala", "Signum", "Tanh", "Treapta"],
    )
    t = st.number_input("Pragul (t):", value=0.0)

    g = 1.0
    if act in ["Liniara", "Sigmoidala", "Tanh"]:
        g = st.number_input("Factorul de scalare (g):", value=1.0)

    if act == "Liniara":
        y = act_lin(t, z, g)
    elif act == "Sigmoidala":
        y = act_sigmoidala(z, t, g)
    elif act == "Signum":
        y = act_signum(z, t)
    elif act == "Tanh":
        y = act_tanh(z, t, g)
    elif act == "Treapta":
        y = act_treapta(z, t)

    tip_iesire = st.radio("Tipul iesirii:", ["Real", "Binar"])

    if tip_iesire == "Binar":
        if act in ["Sigmoidala", "Treapta"]:
            y_final = 1 if y >= 0.5 else 0
        else:
            y_final = 1 if y >= 0 else -1
    else:
        y_final = y

    st.success(f"Valoarea iesirii (y): {y_final}")

    st.divider()
    st.subheader("Graficul functiei de activare")

    z_grafic = np.linspace(-10, 10, 100)
    if act == "Liniara":
        y_grafic = act_lin(t, z_grafic, g)
    elif act == "Sigmoidala":
        y_grafic = act_sigmoidala(z_grafic, t, g)
    elif act == "Signum":
        y_grafic = act_signum(z_grafic, t)
    elif act == "Tanh":
        y_grafic = act_tanh(z_grafic, t, g)
    elif act == "Treapta":
        y_grafic = act_treapta(z_grafic, t)

    df_grafic = pd.DataFrame({"Iesire (y)": y_grafic}, index=z_grafic)
    st.line_chart(df_grafic)


if __name__ == "__main__":
    main()
