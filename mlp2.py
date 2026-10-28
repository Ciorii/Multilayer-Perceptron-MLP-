import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


def calc_gin(intrari, w, metoda):
    valori = np.array(intrari) * np.array(w)
    if metoda == "sum":
        return np.sum(valori)
    if metoda == "prod":
        return np.prod(valori)
    if metoda == "min":
        return np.min(valori)
    if metoda == "max":
        return np.max(valori)


def act_lin(z, t, g):
    return g * (z - t)


def act_sigmoidala(z, t, g):
    return 1 / (1 + np.exp(-g * (z - t)))


def act_signum(z, t):
    return np.sign(z - t)


def act_tanh(z, t, g):
    return np.tanh(g * (z - t))


def act_treapta(z, t):
    return 1 if z >= t else 0


def calc_activare(act, z, t, g):
    if act == "Liniara":
        return act_lin(z, t, g)
    if act == "Sigmoidala":
        return act_sigmoidala(z, t, g)
    if act == "Signum":
        return act_signum(z, t)
    if act == "Tanh":
        return act_tanh(z, t, g)
    return act_treapta(z, t)


def calc_iesire(y, act, tip_iesire):
    if tip_iesire == "Real":
        return y
    if act in ["Sigmoidala", "Treapta"]:
        return 1 if y >= 0.5 else 0
    return 1 if y >= 0 else -1


def calculeaza_strat(intrari, numar_neuroni, strat, setari):
    rezultate = []
    st.markdown(f"#### Calcul pentru stratul {strat}")

    for neuron in range(numar_neuroni):
        w = []
        with st.popover(f"Neuron {neuron + 1}"):
            st.write(f"Setari pentru neuronul {neuron + 1}")
            coloane = st.columns(len(intrari))

            for intrare in range(len(intrari)):
                with coloane[intrare]:
                    w.append(
                        st.number_input(
                            f"w{intrare + 1}",
                            value=0.00,
                            key=f"w_{strat}_{neuron}_{intrare}",
                        )
                    )

            gin = calc_gin(intrari, w, setari["metoda"])
            activare = calc_activare(
                setari["functie"],
                gin,
                setari["t"],
                setari["g"],
            )
            iesire = calc_iesire(
                activare,
                setari["functie"],
                setari["tip"],
            )

            st.write(f"GIN: {gin:.3f}")
            st.write(f"Activare: {activare:.3f}")
            st.write(f"Iesire: {iesire:.3f}")

        rezultate.append(iesire)

    return rezultate


def deseneaza_retea(straturi, valori):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    pozitii = []

    for numar_neuroni in straturi:
        pozitii.append(
            (numar_neuroni - 1) / 2 - np.arange(numar_neuroni)
        )

    for strat in range(len(straturi) - 1):
        for y1 in pozitii[strat]:
            for y2 in pozitii[strat + 1]:
                ax.plot(
                    [strat, strat + 1],
                    [y1, y2],
                    color="yellow",
                    linewidth=0.7,
                    zorder=1,
                )

    for strat, yuri in enumerate(pozitii):
        ax.scatter(
            [strat] * len(yuri),
            yuri,
            s=700,
            color="#2f75b5",
            edgecolors="white",
            zorder=2,
        )

        for neuron, y in enumerate(yuri):
            ax.text(
                strat,
                y,
                f"{valori[strat][neuron]:.2f}",
                color="white",
                ha="center",
                va="center",
            )

    nume_straturi = ["Intrare"]
    nume_straturi += [
        f"Hidden {i}" for i in range(1, len(straturi) - 1)
    ]
    nume_straturi.append("Iesire")

    limita = (max(straturi) - 1) / 2 + 1
    for i, nume in enumerate(nume_straturi):
        ax.text(
            i,
            limita,
            nume,
            color="white",
            ha="center",
            fontweight="bold",
        )

    ax.set_xlim(-0.5, len(straturi) - 0.5)
    ax.set_ylim(-limita, limita + 0.5)
    ax.axis("off")
    return fig


def main():
    st.set_page_config(page_title="Retea neuronala", layout="wide")

    with st.sidebar:
        st.header("Configurare retea")

        neuroni_intrare = st.number_input(
            "Neuroni strat intrare:",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )

        straturi_ascunse = st.number_input(
            "Numar straturi ascunse:",
            min_value=1,
            max_value=3,
            value=1,
            step=1,
        )

        neuroni_iesire = st.number_input(
            "Neuroni strat iesire:",
            min_value=1,
            max_value=20,
            value=1,
            step=1,
        )

        st.subheader("Valori de intrare")
        valori_intrare = []
        for i in range(int(neuroni_intrare)):
            valori_intrare.append(
                st.number_input(
                    f"Intrare {i + 1}:",
                    value=0.0,
                    key=f"valoare_intrare_{i}",
                )
            )

    st.title("Retea neuronala")

    st.subheader("Configurare hidden layer")
    for i in range(1, int(straturi_ascunse) + 1):
        coloana_nume, coloana_neuroni, coloana_libera = st.columns([1, 1, 3])
        with coloana_nume:
            st.markdown(f"**Hidden layer {i}**")
        with coloana_neuroni:
            st.number_input(
                f"Numar neuroni hidden {i}:",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key=f"neuroni_hidden_{i}",
            )

    straturi = [int(neuroni_intrare)]
    straturi += [
        int(st.session_state.get(f"neuroni_hidden_{i}", 1))
        for i in range(1, int(straturi_ascunse) + 1)
    ]
    straturi.append(int(neuroni_iesire))

    valori_straturi = [valori_intrare]

    for strat in range(1, len(straturi)):
        if strat < len(straturi) - 1:
            prefix = str(strat)
        else:
            prefix = "iesire"

        setari = {
            "metoda": st.session_state.get(
                f"metoda_gin_{prefix}",
                "sum",
            ),
            "functie": st.session_state.get(
                f"functie_activare_{prefix}",
                "Liniara",
            ),
            "t": st.session_state.get(
                f"t_{prefix}",
                0.0,
            ),
            "g": st.session_state.get(
                f"g_{prefix}",
                1.0,
            ),
            "tip": st.session_state.get(
                f"tip_iesire_{prefix}",
                "Real",
            ),
        }

        valori_straturi.append(
            calculeaza_strat(
                valori_straturi[-1],
                straturi[strat],
                prefix,
                setari,
            )
        )

    st.subheader("Retea")
    coloane_butoane = st.columns(len(straturi))
    for i in range(1, len(straturi) - 1):
        with coloane_butoane[i]:
            with st.popover(f"ACT + Output {i}"):
                st.write(f"Hidden layer {i}")

                st.selectbox(
                    "Calcul GIN:",
                    ["sum", "prod", "min", "max"],
                    key=f"metoda_gin_{i}",
                )

                functie = st.selectbox(
                    "Functie de activare:",
                    ["Liniara", "Sigmoidala", "Signum", "Tanh", "Treapta"],
                    key=f"functie_activare_{i}",
                )

                st.number_input("t:", value=0.0, key=f"t_{i}")

                if functie in ["Liniara", "Sigmoidala", "Tanh"]:
                    st.number_input(
                        "g:",
                        value=1.0,
                        key=f"g_{i}",
                    )

                st.radio(
                    "Iesire:",
                    ["Real", "Binar"],
                    key=f"tip_iesire_{i}",
                )

    with coloane_butoane[-1]:
        with st.popover("ACT + Output iesire"):
            st.write("Strat de iesire")

            st.selectbox(
                "Calcul GIN:",
                ["sum", "prod", "min", "max"],
                key="metoda_gin_iesire",
            )

            functie_iesire = st.selectbox(
                "Functie de activare:",
                ["Liniara", "Sigmoidala", "Signum", "Tanh", "Treapta"],
                key="functie_activare_iesire",
            )

            st.number_input("t:", value=0.0, key="t_iesire")

            if functie_iesire in ["Liniara", "Sigmoidala", "Tanh"]:
                st.number_input(
                    "g:",
                    value=1.0,
                    key="g_iesire",
                )

            st.radio(
                "Iesire:",
                ["Real", "Binar"],
                key="tip_iesire_iesire",
            )

    fig = deseneaza_retea(straturi, valori_straturi)
    st.pyplot(fig)
    plt.close(fig)

    with st.container(border=True):
        st.subheader("Rezultatul final")
        coloane_rezultat = st.columns(len(valori_straturi[-1]))
        for i, valoare in enumerate(valori_straturi[-1]):
            with coloane_rezultat[i]:
                st.metric(
                    f"Neuron iesire {i + 1}",
                    f"{float(valoare):.3f}",
                )


if __name__ == "__main__":
    main()
