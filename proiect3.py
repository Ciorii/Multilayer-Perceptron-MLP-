import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import csv
import re


COLOANE_INTRARE = [
    "CpuNumberOfCores",
    "CpuNumberOfThreads",
    "CpuTurboClock",
    "GpuBandwidth",
    "GpuBoostClock",
    "GpuMemorySize",
    "GameName",
    "GameSetting",
]

COLOANE_CATEGORICE = [
    "GameName",
    "GameSetting",
]

COLOANA_TINTA = "FPS"
FPS_MINIM = 0
FPS_MAXIM = 700

NUME_COLOANE = {
    "CpuNumberOfCores": "Numar de coruri CPU",
    "CpuNumberOfThreads": "Numar de thread-uri CPU",
    "CpuTurboClock": "Frecventa Boost CPU (GHz)",
    "GpuBandwidth": "Bandwidth GPU (GB/s)",
    "GpuBoostClock": "Frecventa Boost GPU (MHz)",
    "GpuMemorySize": "VRAM GPU (GB)",
    "GameName": "Joc",
    "GameSetting": "Setare grafica",
}

LIMITE_HARDWARE = {
    "CpuNumberOfCores": (1, 32),
    "CpuNumberOfThreads": (1, 64),
    "CpuTurboClock": (1000, 6000),
    "GpuBandwidth": (1000, 2000000),
    "GpuBoostClock": (300, 3000),
    "GpuMemorySize": (128, 48000),
}

def calc_gin(x, w):
    return np.dot(x, w)


def act_tanh(z, t, g):
    return np.tanh(g * (z - t))


def derivata_tanh(z, t, g):
    y = act_tanh(z, t, g)
    return g * (1 - y ** 2)


def act_sigmoid(z):
    z = np.clip(z, -60, 60)
    return 1 / (1 + np.exp(-z))


def derivata_sigmoid(z):
    y = act_sigmoid(z)
    return y * (1 - y)


def calc_activare(z, functie, t, g):
    if functie == "Tanh":
        return act_tanh(z, t, g)
    return act_sigmoid(z)


def calc_derivata(z, functie, t, g):
    if functie == "Tanh":
        return derivata_tanh(z, t, g)
    return derivata_sigmoid(z)


def initializeaza_ponderi(straturi):
    w = []

    for strat in range(len(straturi) - 1):
        numar_intrari = straturi[strat]
        numar_neuroni = straturi[strat + 1]

        w.append(
            np.random.uniform(
                -1,
                1,
                (numar_intrari, numar_neuroni),
            ).astype(np.float32)
        )

    return w


def forward_propagation(x, w, functii_activare, t, g):
    y = [np.array(x, dtype=np.float32)]
    z = []

    for index, w_strat in enumerate(w):
        z_strat = calc_gin(y[-1], w_strat)
        y_strat = calc_activare(
            z_strat,
            functii_activare[index],
            t,
            g,
        )

        z.append(z_strat)
        y.append(y_strat)

    return y, z


def calc_eroare(t, y):
    return 0.5 * (y - t) ** 2


def actualizeaza_ponderi(w, dw, rata_invatare):
    w_noi = []

    for w_strat, gradient in zip(w, dw):
        w_nou = w_strat - rata_invatare * gradient
        w_noi.append(w_nou)

    return w_noi


def antreneaza_retea(
    x,
    t,
    straturi,
    functii_activare,
    rata_invatare,
    numar_epoci,
    eroare_acceptata,
    t_prag,
    g,
    progres=None,
):
    w = initializeaza_ponderi(straturi)
    erori = []

    for epoca in range(numar_epoci):
        y, z = forward_propagation(
            x,
            w,
            functii_activare,
            t_prag,
            g,
        )
        eroare = calc_eroare(t, y[-1])
        eroare_epoca = float(np.mean(eroare))

        delta = [None] * len(w)
        delta[-1] = (
            (y[-1] - t)
            * calc_derivata(
                z[-1],
                functii_activare[-1],
                t_prag,
                g,
            )
        )

        for strat in range(len(w) - 2, -1, -1):
            delta[strat] = (
                np.dot(delta[strat + 1], w[strat + 1].T)
                * calc_derivata(
                    z[strat],
                    functii_activare[strat],
                    t_prag,
                    g,
                )
            )

        dw = []
        for strat in range(len(w)):
            gradient = np.dot(y[strat].T, delta[strat])
            gradient = gradient / len(x)
            dw.append(gradient)

        w = actualizeaza_ponderi(w, dw, rata_invatare)
        erori.append(eroare_epoca)

        if progres is not None:
            progres(epoca + 1, numar_epoci, eroare_epoca)

        if eroare_epoca <= eroare_acceptata:
            break

    return w, erori


def prezice(x, w, functii_activare, t_prag, g):
    y, _ = forward_propagation(
        x,
        w,
        functii_activare,
        t_prag,
        g,
    )
    return y[-1]


def normalizeaza_date(x):
    x = np.asarray(x, dtype=np.float32)
    x_min = np.min(x, axis=0).astype(np.float32)
    x_max = np.max(x, axis=0).astype(np.float32)
    diferenta = x_max - x_min
    diferenta[diferenta == 0] = 1
    x_normalizat = ((x - x_min) / diferenta).astype(np.float32)
    return x_normalizat, x_min, x_max


def normalizeaza_hardware(x):
    x = np.asarray(x, dtype=np.float32)
    minim = np.array(
        [LIMITE_HARDWARE[coloana][0] for coloana in COLOANE_INTRARE
         if coloana not in COLOANE_CATEGORICE],
        dtype=np.float32,
    )
    maxim = np.array(
        [LIMITE_HARDWARE[coloana][1] for coloana in COLOANE_INTRARE
         if coloana not in COLOANE_CATEGORICE],
        dtype=np.float32,
    )
    return (x - minim) / (maxim - minim), minim, maxim


def denormalizeaza_date(x, x_min, x_max):
    return x * (x_max - x_min) + x_min


def denormalizeaza_fps(y, t_min, t_max):
    y = np.clip(y, 0, 1)
    return denormalizeaza_date(y, t_min, t_max)


def imparte_datele(x, t, procent_antrenare):
    ordine = np.random.permutation(len(x))
    limita = int(len(x) * procent_antrenare)

    ordine_antrenare = ordine[:limita]
    ordine_testare = ordine[limita:]

    return (
        x[ordine_antrenare],
        t[ordine_antrenare],
        x[ordine_testare],
        t[ordine_testare],
    )


def citeste_arff(fisier):
    continut = fisier.read()
    if isinstance(continut, bytes):
        continut = continut.decode("utf-8", errors="replace")

    linii = continut.splitlines()
    coloane = []
    index_date = 0

    for index, linie in enumerate(linii):
        potrivire = re.match(
            r"@attribute\s+'([^']+)'",
            linie.strip(),
            re.IGNORECASE,
        )
        if potrivire:
            coloane.append(potrivire.group(1))
        if linie.strip().lower() == "@data":
            index_date = index + 1
            break

    date = []
    for linie in linii[index_date:]:
        if linie.strip() and not linie.lstrip().startswith("%"):
            date.append(next(csv.reader([linie])))

    return pd.DataFrame(date, columns=coloane)


def citeste_fisier(fisier):
    if fisier.name.lower().endswith(".arff"):
        return citeste_arff(fisier)
    return pd.read_csv(fisier, na_values=["?"])


def curata_datele(df):
    coloane = COLOANE_INTRARE + [COLOANA_TINTA]
    df = df[coloane].copy()
    df = df.replace("?", np.nan)

    numerice = [
        coloana
        for coloana in coloane
        if coloana not in COLOANE_CATEGORICE
    ]

    for coloana in numerice:
        df[coloana] = pd.to_numeric(
            df[coloana],
            errors="coerce",
        )

    df = df[
        (df[COLOANA_TINTA] > FPS_MINIM)
        & (df[COLOANA_TINTA] <= FPS_MAXIM)
    ]
    df = df.dropna()

    return df


def pregateste_datele(df):
    categorice = COLOANE_CATEGORICE
    codificari = {}
    numerice = [
        coloana for coloana in COLOANE_INTRARE
        if coloana not in categorice
    ]
    valori_x = [
        df[coloana].to_numpy(dtype=np.float32)
        for coloana in numerice
    ]
    for coloana in categorice:
        valori = df[coloana].astype(str).unique().tolist()
        if coloana == "GameSetting":
            ordine = ["med", "max"]
            valori = [
                valoare
                for valoare in ordine
                if valoare in valori
            ]
        else:
            valori = sorted(valori)

        codificari[coloana] = {
            valoare: index
            for index, valoare in enumerate(valori)
        }
        for valoare in valori:
            valori_x.append(
                (
                    df[coloana].astype(str) == valoare
                ).to_numpy(dtype=np.float32)
            )

    x = np.column_stack(valori_x)
    t = df[COLOANA_TINTA].to_numpy(dtype=np.float32).reshape(-1, 1)

    numar_numerice = len(numerice)
    x_numerice, x_min, x_max = normalizeaza_hardware(
        x[:, :numar_numerice]
    )
    x_categorice = x[:, numar_numerice:]
    x = np.column_stack([x_numerice, x_categorice])
    x_min = np.concatenate(
        [x_min, np.zeros(x_categorice.shape[1], dtype=np.float32)]
    )
    x_max = np.concatenate(
        [x_max, np.ones(x_categorice.shape[1], dtype=np.float32)]
    )
    t, t_min, t_max = normalizeaza_date(t)

    return (
        x,
        t,
        x_min,
        x_max,
        t_min,
        t_max,
        codificari,
    )


def calculeaza_metrici(t_real, t_pred):
    eroare = t_pred - t_real
    mae = np.mean(np.abs(eroare))
    return mae


def main():
    st.set_page_config(
        page_title="Estimarea FPS",
        layout="wide",
    )

    st.title("Estimarea FPS cu o retea neuronala")
    st.info(
        "W initializate in [-1, 1]."
    )

    fisier = st.file_uploader(
        "Incarca fisierul CSV sau ARFF",
        type=["csv", "arff"],
    )
    if fisier is None:
        st.warning("Incarca un fisier CSV pentru a continua.")
        return

    df = citeste_fisier(fisier)
    st.subheader("Datele din fisier")
    st.write(f"Numar randuri: {len(df)}")
    coloane_afisate = COLOANE_INTRARE + [COLOANA_TINTA]
    st.dataframe(df[coloane_afisate].head())

    coloane_lipsa = [
        coloana
        for coloana in (
            COLOANE_INTRARE
            + [COLOANA_TINTA]
        )
        if coloana not in df.columns
    ]
    if coloane_lipsa:
        st.error(
            "Lipsesc urmatoarele coloane din fisier: "
            + ", ".join(coloane_lipsa)
        )
        return

    df = curata_datele(df)
    if not pd.api.types.is_numeric_dtype(df[COLOANA_TINTA]):
        st.error("Coloana tinta trebuie sa contina valori numerice.")
        return

    st.info("Se folosesc toate jocurile si toate valorile valide din fisier.")

    (
        x,
        t,
        x_min,
        x_max,
        t_min,
        t_max,
        codificari,
    ) = pregateste_datele(df)
    coloane_categorice = list(codificari.keys())

    if len(x) < 2:
        st.error("Nu exista suficiente date pentru antrenare si testare.")
        return

    st.info(
        f"Date pregatite: {len(x)} randuri, "
        f"{len(codificari['GameName'])} jocuri, "
        f"{len(codificari['GameSetting'])} setari grafice."
    )

    st.sidebar.header("Configurare retea")
    numar_straturi = st.sidebar.number_input(
        "Numar straturi ascunse:",
        min_value=1,
        max_value=10,
        value=1,
        step=1,
    )
    neuroni_hidden = []
    for strat in range(int(numar_straturi)):
        neuroni_hidden.append(
            st.sidebar.number_input(
                f"Neuroni in stratul ascuns {strat + 1}:",
                min_value=1,
                max_value=10,
                value=8,
                step=1,
                key=f"neuroni_hidden_{strat}",
            )
        )
    t_prag = 0.0
    g = 1.0
    st.sidebar.info(
        "Functia Tanh foloseste valorile fixe t = 0 si g = 1."
    )
    numar_epoci = st.sidebar.number_input(
        "Numar maxim de epoci:",
        min_value=1,
        max_value=10000,
        value=1000,
        step=100,
    )
    rata_invatare = st.sidebar.number_input(
        "Rata de invatare:",
        min_value=0.00001,
        max_value=1.0,
        value=0.01,
        format="%.5f",
    )
    eroare_acceptata = st.sidebar.number_input(
        "Eroare acceptata:",
        min_value=0.000001,
        value=0.001,
        format="%.6f",
    )

    if st.button("Antreneaza reteaua"):
        status = st.empty()
        bara_progres = st.progress(0)
        grafic_progres = st.empty()
        erori_live = []

        def afiseaza_progres(epoca, total_epoci, eroare):
            erori_live.append(eroare)
            bara_progres.progress(epoca / total_epoci)
            status.info(
                f"Antrenare: epoca {epoca}/{total_epoci} | "
                f"eroare: {eroare:.8f}"
            )
            fig_live, ax_live = plt.subplots()
            ax_live.plot(
                np.arange(1, len(erori_live) + 1),
                erori_live,
            )
            ax_live.set_xlabel("Epoca")
            ax_live.set_ylabel("Eroare")
            ax_live.set_xlim(left=1)
            ax_live.grid()
            grafic_progres.pyplot(fig_live)
            plt.close(fig_live)

        x_antrenare, t_antrenare, x_testare, t_testare = imparte_datele(
            x,
            t,
            0.8,
        )

        straturi = [
            x.shape[1],
            *[int(nr) for nr in neuroni_hidden],
            1,
        ]
        functii_activare = (
            ["Tanh"] * int(numar_straturi)
            + ["Sigmoid"]
        )
        w, erori = antreneaza_retea(
            x_antrenare,
            t_antrenare,
            straturi,
            functii_activare,
            rata_invatare,
            int(numar_epoci),
            eroare_acceptata,
            t_prag,
            g,
            afiseaza_progres,
        )

        bara_progres.progress(1.0)
        status.success(
            f"Antrenarea s-a terminat in {len(erori)} epoci."
        )
        st.session_state["w"] = w
        st.session_state["erori"] = erori
        st.session_state["x_testare"] = x_testare
        st.session_state["t_testare"] = t_testare
        st.session_state["t_min"] = t_min
        st.session_state["t_max"] = t_max
        st.session_state["x_min"] = x_min
        st.session_state["x_max"] = x_max
        st.session_state["coloane_intrare"] = COLOANE_INTRARE
        st.session_state["coloane_categorice"] = coloane_categorice
        st.session_state["codificari"] = codificari
        st.session_state["t_prag"] = t_prag
        st.session_state["g"] = g
        st.session_state["functii_activare"] = functii_activare

    st.sidebar.header("Model")
    model_incarcat = st.sidebar.file_uploader(
        "Incarca modelul:",
        type=["pkl"],
    )
    if model_incarcat is not None and st.sidebar.button(
        "Foloseste modelul incarcat"
    ):
        try:
            model = pickle.load(model_incarcat)
            chei_obligatorii = [
                "w",
                "erori",
                "x_testare",
                "t_testare",
                "t_min",
                "t_max",
                "x_min",
                "x_max",
                "codificari",
                "coloane_intrare",
                "coloane_categorice",
                "t_prag",
                "g",
                "functii_activare",
            ]
            if not all(cheie in model for cheie in chei_obligatorii):
                raise ValueError("Modelul nu contine toate datele necesare.")

            for cheie, valoare in model.items():
                st.session_state[cheie] = valoare
            st.sidebar.success("Modelul a fost incarcat.")
        except (pickle.UnpicklingError, EOFError, ValueError, KeyError):
            st.sidebar.error("Fisierul nu este un model valid.")

    if "w" not in st.session_state:
        return

    model = {
        "w": st.session_state["w"],
        "erori": st.session_state["erori"],
        "x_testare": st.session_state["x_testare"],
        "t_testare": st.session_state["t_testare"],
        "t_min": st.session_state["t_min"],
        "t_max": st.session_state["t_max"],
        "x_min": st.session_state["x_min"],
        "x_max": st.session_state["x_max"],
        "codificari": st.session_state["codificari"],
        "coloane_intrare": st.session_state["coloane_intrare"],
        "coloane_categorice": st.session_state["coloane_categorice"],
        "t_prag": st.session_state["t_prag"],
        "g": st.session_state["g"],
        "functii_activare": st.session_state["functii_activare"],
    }
    st.sidebar.download_button(
        "Salveaza modelul",
        data=pickle.dumps(model),
        file_name="model_fps.pkl",
        mime="application/octet-stream",
    )

    erori = st.session_state["erori"]
    epoci = np.arange(1, len(erori) + 1)

    with st.container(border=True):
        st.markdown("#### Valorile exacte ale erorii")
        coloana_curenta, coloana_initiala, coloana_minima = st.columns(3)
        coloana_curenta.metric(
            "Eroarea curenta",
            f"{erori[-1]:.8f}",
            f"{erori[-1] - erori[0]:.8f}",
        )
        coloana_initiala.metric(
            "Eroarea initiala",
            f"{erori[0]:.8f}",
        )
        coloana_minima.metric(
            "Cea mai mica eroare",
            f"{min(erori):.8f}",
            f"Epoca {int(np.argmin(erori) + 1)}",
        )

        tabel_erori = pd.DataFrame(
            {
                "Epoca": epoci,
                "Eroare": erori,
            }
        )
        st.dataframe(
            tabel_erori,
            use_container_width=True,
            height=220,
        )

    x_testare = st.session_state["x_testare"]
    t_testare = st.session_state["t_testare"]
    w = st.session_state["w"]

    predictii = prezice(
        x_testare,
        w,
        st.session_state["functii_activare"],
        st.session_state["t_prag"],
        st.session_state["g"],
    ).reshape(-1, 1)
    t_testare_real = denormalizeaza_date(
        t_testare,
        st.session_state["t_min"],
        st.session_state["t_max"],
    )
    predictii_reale = denormalizeaza_fps(
        predictii,
        st.session_state["t_min"],
        st.session_state["t_max"],
    )

    mae = calculeaza_metrici(
        t_testare_real,
        predictii_reale,
    )

    st.subheader("Rezultatele testarii")
    st.metric("MAE (FPS)", f"{mae:.2f}")

    rezultate = pd.DataFrame(
        {
            "FPS real": t_testare_real.flatten(),
            "FPS prezis": predictii_reale.flatten(),
        }
    )
    st.dataframe(rezultate)

    st.subheader("Predictie pentru o configuratie noua")
    st.write(
        "Alege CPU-ul, GPU-ul, jocul si setarea. "
        "FPS-ul este rezultatul retelei."
    )

    valori_noi = {}
    st.write(
        "Caracteristicile hardware se introduc in unitatile afisate. "
        "Programul le transforma in unitatile folosite de dataset."
    )

    setari_hardware = {
        "CpuNumberOfCores": (1, "%.0f"),
        "CpuNumberOfThreads": (1, "%.0f"),
        "CpuTurboClock": (0.1, "%.1f"),
        "GpuBandwidth": (1.0, "%.0f"),
        "GpuBoostClock": (1, "%.0f"),
        "GpuMemorySize": (1.0, "%.0f"),
    }
    unitati_dataset = {
        "CpuTurboClock": 1000,
        "GpuBandwidth": 1000,
        "GpuMemorySize": 1000,
    }
    etichete_unitati = {
        "CpuTurboClock": "Frecventa Boost CPU (GHz)",
        "GpuBandwidth": "Bandwidth GPU (GB/s)",
        "GpuMemorySize": "VRAM GPU (GB)",
    }

    for coloana in COLOANE_INTRARE:
        if coloana in st.session_state["coloane_categorice"]:
            optiuni = list(
                st.session_state["codificari"][coloana].keys()
            )
            valori_noi[coloana] = st.selectbox(
                f"{NUME_COLOANE[coloana]}:",
                optiuni,
                key=f"nou_{coloana}",
            )
            continue

        pas, format_valoare = setari_hardware[coloana]
        factor = unitati_dataset.get(coloana, 1)
        minim = LIMITE_HARDWARE[coloana][0] / factor
        maxim = LIMITE_HARDWARE[coloana][1] / factor
        valoare = (minim + maxim) / 2
        if pas == 1:
            valoare = int(round(valoare))

        eticheta = etichete_unitati.get(
            coloana,
            f"{NUME_COLOANE[coloana]}:",
        )
        valori_noi[coloana] = st.number_input(
            f"{eticheta}:",
            min_value=float(minim),
            max_value=float(maxim),
            value=float(valoare),
            step=float(pas),
            format=format_valoare,
            key=f"nou_{coloana}",
        )

    if st.button("Calculeaza FPS"):
        x_nou = []
        for coloana in st.session_state["coloane_intrare"]:
            if coloana in st.session_state["coloane_categorice"]:
                valoare = str(valori_noi[coloana])
                x_nou.extend(
                    [
                        float(
                            valoare
                            == categorie
                        )
                        for categorie in st.session_state[
                            "codificari"
                        ][coloana]
                    ]
                )
            else:
                valoare = float(valori_noi[coloana])
                if coloana in unitati_dataset:
                    valoare *= unitati_dataset[coloana]
                x_nou.append(valoare)

        x_nou = np.array([x_nou], dtype=np.float32)
        diferenta = (
            st.session_state["x_max"]
            - st.session_state["x_min"]
        )
        diferenta[diferenta == 0] = 1
        x_nou = (
            x_nou - st.session_state["x_min"]
        ) / diferenta

        y_nou = prezice(
            x_nou[0],
            w,
            st.session_state["functii_activare"],
            st.session_state["t_prag"],
            st.session_state["g"],
        )
        fps_nou = denormalizeaza_fps(
            y_nou,
            st.session_state["t_min"],
            st.session_state["t_max"],
        )
        st.success(f"FPS estimat: {float(fps_nou[0]):.2f}")


if __name__ == "__main__":
    main()