# streamlit_app.py
import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

st.set_page_config(page_title="Aksjesimulering", layout="wide")

# Standarddata (kan lastes/saves til st.session_state for enkel persistering)
default_stocks = [
    {"name": "Fisk", "min_change": -5.0, "max_change": 5.0, "value": 100.0},
    {"name": "Gull", "min_change": -2.0, "max_change": 2.0, "value": 100.0},
    {"name": "IT",   "min_change": -10.0, "max_change": 10.0, "value": 100.0},
    {"name": "Olje", "min_change": -7.0,  "max_change": 7.0,  "value": 100.0},
]
default_events = [
    {"name": "Storm", "probability": 10.0, "impact": -5.0, "affected_stocks": ["Fisk", "Olje"]},
    {"name": "Teknologisk fremskritt", "probability": 5.0, "impact": 10.0, "affected_stocks": ["IT"]},
]

if "stocks" not in st.session_state:
    st.session_state.stocks = default_stocks.copy()
if "events" not in st.session_state:
    st.session_state.events = default_events.copy()

st.title("Aksjesimulering (web)")

colL, colR = st.columns([1,1])

with colL:
    st.subheader("Spillinnstillinger")
    total_weeks = st.number_input("Antall uker", min_value=1, max_value=52, value=5, step=1)
    max_events_per_week = st.number_input("Maks hendelser per uke", min_value=0, max_value=10, value=1, step=1)

    st.subheader("Aksjer")
    # Enkel editor for min/max/value
    for s in st.session_state.stocks:
        with st.expander(s["name"], expanded=False):
            s["min_change"] = st.number_input(f"Min % ({s['name']})", value=float(s["min_change"]), step=0.1, key=f"min_{s['name']}")
            s["max_change"] = st.number_input(f"Maks % ({s['name']})", value=float(s["max_change"]), step=0.1, key=f"max_{s['name']}")
            s["value"]      = st.number_input(f"Startverdi ({s['name']})", value=float(s["value"]), step=1.0, key=f"val_{s['name']}")

    st.subheader("Hendelser")
    for e in st.session_state.events:
        with st.expander(e["name"], expanded=False):
            e["probability"] = st.number_input(f"Sannsynlighet % ({e['name']})", value=float(e["probability"]), step=0.5, key=f"prob_{e['name']}")
            e["impact"]      = st.number_input(f"Påvirkning % ({e['name']})", value=float(e["impact"]), step=0.5, key=f"imp_{e['name']}")
            # Affected: enkel avkryssing
            picks = []
            for s in st.session_state.stocks:
                checked = st.checkbox(f"Påvirker {s['name']}", value=(s["name"] in e["affected_stocks"]), key=f"aff_{e['name']}_{s['name']}")
                if checked: picks.append(s["name"])
            e["affected_stocks"] = picks

start_sim = colR.button("Start simulering")

if start_sim:
    # Kjør 1 uke som eksempel – du kan loope total_weeks hvis ønskelig
    stocks = [dict(s) for s in st.session_state.stocks]
    initial_values = {s["name"]: s["value"] for s in stocks}
    previous_values = {s["name"]: s["value"] for s in stocks}

    total_steps = 30
    days = ["Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag"]
    stock_histories = {s["name"]: [s["value"]] for s in stocks}
    active_events = []
    new_events_this_week = 0

    for step in range(total_steps):
        # Prøv å trigge ny hendelse
        if new_events_this_week < max_events_per_week:
            for ev in st.session_state.events:
                if random.randint(1, 100) <= int(round(ev["probability"])):
                    active_events.append([ev, 3])  # varer 3 steg
                    new_events_this_week += 1
                    break

        # Beregn endringer
        for s in stocks:
            total_change = random.uniform(s["min_change"], s["max_change"])
            for ev, steps_left in active_events:
                if s["name"] in ev["affected_stocks"]:
                    total_change += ev["impact"]

            normalized = max(min(s["value"], 300), 20)
            adjustment_factor = 100 / normalized
            adjusted_change = total_change * adjustment_factor

            s["value"] = max(s["value"] * (1 + adjusted_change/100), 5)
            stock_histories[s["name"]].append(s["value"])

        # Tikk ned aktive hendelser
        for ev in active_events:
            ev[1] -= 1
        active_events = [ev for ev in active_events if ev[1] > 0]

    # Plot
    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_title("Aksjeutvikling (1 uke)")
    ax.set_xlabel("Tid")
    ax.set_ylabel("Verdi")
    ax.set_xlim(0, total_steps)
    ax.set_xticks([i * 6 for i in range(len(days))])
    ax.set_xticklabels(days)
    for name, values in stock_histories.items():
        ax.plot(range(len(values)), values, label=name)
    ax.legend()
    st.pyplot(fig)

    # Oppsummering
    st.subheader("Oppsummering uke")
    for s in stocks:
        name = s["name"]
        new_price = s["value"]
        last_price = previous_values[name]
        start_price = initial_values[name]
        week_change = ((new_price - last_price) / last_price) * 100
        total_change = ((new_price - start_price) / start_price) * 100
        st.write(f"**{name}** – Uke: {week_change:+.2f}%  |  Total: {total_change:+.2f}%  |  Ny pris: {new_price:.2f} kr")
