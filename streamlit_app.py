# streamlit_app.py
import io
import json
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ---------------------- Sideoppsett ----------------------
st.set_page_config(
    page_title="Aksjesimulering (web)",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Litt “app look”
st.markdown(
    """
    <style>
    .block-container {padding-top: 1.2rem; padding-bottom: 2rem;}
    .stButton > button {height: 3rem; font-weight: 600;}
    .metric-green {color: #107C10;}
    .metric-red {color: #CF1322;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- Datatyper ----------------------
@dataclass
class Stock:
    name: str
    min_change: float
    max_change: float
    value: float


@dataclass
class Event:
    name: str
    probability: float  # 0-100
    impact: float       # +/- prosent
    affected_stocks: List[str]


# ---------------------- Standarddata ----------------------
DEFAULT_STOCKS = [
    Stock("Fisk", -5.0, 5.0, 100.0),
    Stock("Gull", -2.0, 2.0, 100.0),
    Stock("IT", -10.0, 10.0, 100.0),
    Stock("Olje", -7.0, 7.0, 100.0),
]

DEFAULT_EVENTS = [
    Event("Storm", 10.0, -5.0, ["Fisk", "Olje"]),
    Event("Teknologisk fremskritt", 5.0, 10.0, ["IT"]),
]


# ---------------------- Hjelp ----------------------
def to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def ensure_unique_names(items: List[str]) -> bool:
    """Returnerer True hvis alle navn er unike og ikke tomme."""
    names = [n.strip() for n in items if str(n).strip() != ""]
    return len(names) == len(set(names)) and len(names) > 0


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------- Init session_state ----------------------
if "stocks" not in st.session_state:
    st.session_state.stocks: List[Stock] = [Stock(**asdict(s)) for s in DEFAULT_STOCKS]
if "events" not in st.session_state:
    st.session_state.events: List[Event] = [Event(**asdict(e)) for e in DEFAULT_EVENTS]

# Sim-tilstand
if "total_weeks" not in st.session_state:
    st.session_state.total_weeks: int = 5
if "max_events_per_week" not in st.session_state:
    st.session_state.max_events_per_week: int = 1
if "seed" not in st.session_state:
    st.session_state.seed = None  # eller et tall for deterministisk

if "current_week" not in st.session_state:
    st.session_state.current_week: int = 0
if "initial_values" not in st.session_state:
    st.session_state.initial_values: Dict[str, float] = {}
if "previous_values" not in st.session_state:
    st.session_state.previous_values: Dict[str, float] = {}
if "histories" not in st.session_state:
    st.session_state.histories: Dict[str, List[float]] = {}
if "weekly_summaries" not in st.session_state:
    # Liste av (uke, rows) der rows = List[Tuple[name, uke%, total%, ny pris]]
    st.session_state.weekly_summaries: List[Tuple[int, List[Tuple[str, float, float, float]]]] = []
if "last_fig_png" not in st.session_state:
    st.session_state.last_fig_png: bytes = b""
if "event_log" not in st.session_state:
    st.session_state.event_log: List[str] = []


def reset_simulation():
    st.session_state.current_week = 0
    st.session_state.initial_values = {}
    st.session_state.previous_values = {}
    st.session_state.histories = {}
    st.session_state.weekly_summaries = []
    st.session_state.last_fig_png = b""
    st.session_state.event_log = []


# ---------------------- Sidebar (Innstillinger) ----------------------
with st.sidebar:
    st.header("⚙️ Spillinnstillinger")

    st.session_state.total_weeks = st.number_input(
        "Antall uker",
        min_value=1, max_value=52, value=st.session_state.total_weeks, step=1,
    )

    st.session_state.max_events_per_week = st.number_input(
        "Maks hendelser per uke",
        min_value=0, max_value=10, value=st.session_state.max_events_per_week, step=1,
    )

    st.session_state.seed = st.text_input(
        "Tilfeldig frø (tomt = mer tilfeldig)", value=str(st.session_state.seed or "")
    ) or None

    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔁 Nullstill simulering", use_container_width=True):
            reset_simulation()
            st.toast("Simulering nullstilt.", icon="♻️")
    with c2:
        if st.button("↩️ Tilbakestill data", use_container_width=True):
            st.session_state.stocks = [Stock(**asdict(s)) for s in DEFAULT_STOCKS]
            st.session_state.events = [Event(**asdict(e)) for e in DEFAULT_EVENTS]
            reset_simulation()
            st.toast("Aksjer og hendelser tilbakestilt.", icon="✅")

    st.markdown("---")
    st.caption("Data lagres i nettleserøkten (Session State).")


# ---------------------- Redigeringsfaner ----------------------
st.title("📈 Aksjesimulering (web)")

tab_dash, tab_edit, tab_results = st.tabs(["Dashboard", "Aksjer & Hendelser", "Resultater"])

# ---------------------- Tab: Dashboard ----------------------
with tab_dash:
    st.subheader("Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Uke (nåværende)", st.session_state.current_week)
    c2.metric("Antall uker (mål)", st.session_state.total_weeks)
    c3.metric("Maks hendelser / uke", st.session_state.max_events_per_week)
    c4.metric("Antall aksjer", len(st.session_state.stocks))

    st.markdown("### Siste uke: graf og sammendrag")
    if st.session_state.weekly_summaries:
        wk, rows = st.session_state.weekly_summaries[-1]

        # Siste figur som PNG (hvis lagret)
        if st.session_state.last_fig_png:
            st.image(st.session_state.last_fig_png, caption=f"Aksjeutvikling – uke {wk}", use_container_width=True)

        # Tabell for ukesammendrag
        df = pd.DataFrame(rows, columns=["Aksje", "Uke-endring (%)", "Total endring (%)", "Ny pris (kr)"])
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Nedlasting CSV
        csv_bytes = df.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button(
            "⬇️ Last ned ukesammendrag (CSV)",
            data=csv_bytes,
            file_name=f"ukesammendrag_uke{wk}.csv",
            mime="text/csv",
        )
    else:
        st.info("Ingen ukesammendrag ennå. Kjør simulering under.")

    st.markdown("### Kjør simulering")
    cA, cB, cC = st.columns([1,1,2])
    with cA:
        if st.button("▶️ Neste uke"):
            ran = np.random.RandomState(seed=int(st.session_state.seed)) if st.session_state.seed else np.random
            # kjør presis 1 uke
            ok, msg = simulate_one_week(ran)
            if ok:
                st.toast(f"Uke {st.session_state.current_week} fullført.", icon="✅")
            else:
                st.toast(msg, icon="⚠️")
    with cB:
        if st.button("⏩ Kjør alle uker"):
            ran = np.random.RandomState(seed=int(st.session_state.seed)) if st.session_state.seed else np.random
            while st.session_state.current_week < st.session_state.total_weeks:
                ok, msg = simulate_one_week(ran)
                if not ok:
                    st.toast(msg, icon="⚠️")
                    break
            if st.session_state.current_week >= st.session_state.total_weeks:
                st.toast("Simuleringen er ferdig!", icon="🎉")

    with cC:
        if st.session_state.last_fig_png:
            st.download_button(
                "🖼️ Last ned siste graf (PNG)",
                data=st.session_state.last_fig_png,
                file_name="aksjeutvikling.png",
                mime="image/png",
            )

    # Eventlogg
    st.markdown("### Hendelseslogg (siste uke)")
    if st.session_state.event_log:
        for line in st.session_state.event_log[-10:]:
            st.write("• " + line)
    else:
        st.caption("Ingen hendelser ennå.")


# ---------------------- Tab: Aksjer & Hendelser ----------------------
with tab_edit:
    left, right = st.columns(2)

    with left:
        st.subheader("Aksjer")
        st.caption("Rediger direkte i tabellen. Nye rader legges til nederst. Slett rader ved å krysse dem ut.")
        stock_df = pd.DataFrame([asdict(s) for s in st.session_state.stocks])
        # Konfig for penere editor
        stock_cfg = {
            "name": st.column_config.TextColumn("Aksje", width="medium", required=True),
            "min_change": st.column_config.NumberColumn("Min %", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f"),
            "max_change": st.column_config.NumberColumn("Maks %", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f"),
            "value": st.column_config.NumberColumn("Startverdi", min_value=0.0, max_value=1_000_000.0, step=1.0, format="%.2f"),
        }
        edited_stocks = st.data_editor(
            stock_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config=stock_cfg,
            hide_index=True,
            key="stock_editor",
        )

        if st.button("💾 Lagre aksjer"):
            # Valider
            names_ok = ensure_unique_names(edited_stocks["name"].tolist())
            if not names_ok:
                st.error("Aksjenavn må være unike og ikke tomme.")
            else:
                try:
                    new_stocks = []
                    for _, row in edited_stocks.iterrows():
                        new_stocks.append(
                            Stock(
                                name=str(row["name"]).strip(),
                                min_change=float(row["min_change"]),
                                max_change=float(row["max_change"]),
                                value=float(row["value"]),
                            )
                        )
                    st.session_state.stocks = new_stocks
                    st.success("Aksjer lagret.")
                except Exception as e:
                    st.error(f"Kunne ikke lagre aksjer: {e}")

    with right:
        st.subheader("Hendelser")
        st.caption("Rediger sannsynlighet/påvirkning. Marker hvilke aksjer som påvirkes per hendelse.")
        # For enkel editing: vis events i to deler—tabell + multiselect for hver rad
        ev_df = pd.DataFrame(
            [
                {"name": e.name, "probability": e.probability, "impact": e.impact}
                for e in st.session_state.events
            ]
        )
        ev_cfg = {
            "name": st.column_config.TextColumn("Hendelse", width="large", required=True),
            "probability": st.column_config.NumberColumn("Sannsynlighet (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
            "impact": st.column_config.NumberColumn("Påvirkning (%)", min_value=-100.0, max_value=100.0, step=0.5, format="%.2f"),
        }
        edited_ev = st.data_editor(
            ev_df,
            num_rows="dynamic",
            use_container_width=True,
            column_config=ev_cfg,
            hide_index=True,
            key="event_editor_rows",
        )

        # Multiselect per rad
        affected_controls: List[List[str]] = []
        for i in range(len(edited_ev)):
            cols = st.columns([1, 2])
            with cols[0]:
                st.caption(f"Påvirker (rad {i+1})")
            with cols[1]:
                current_name = edited_ev.iloc[i]["name"] if i < len(st.session_state.events) else ""
                preselect = []
                if i < len(st.session_state.events) and current_name == st.session_state.events[i].name:
                    preselect = st.session_state.events[i].affected_stocks
                affected = st.multiselect(
                    label=f"Aksjer (rad {i+1})",
                    options=[s.name for s in st.session_state.stocks],
                    default=preselect,
                    key=f"aff_sel_{i}",
                )
                affected_controls.append(affected)

        if st.button("💾 Lagre hendelser"):
            names_ok = ensure_unique_names(edited_ev["name"].tolist())
            if not names_ok:
                st.error("Hendelsesnavn må være unike og ikke tomme.")
            else:
                try:
                    new_events = []
                    for i, row in edited_ev.iterrows():
                        new_events.append(
                            Event(
                                name=str(row["name"]).strip(),
                                probability=float(row["probability"]),
                                impact=float(row["impact"]),
                                affected_stocks=[str(x) for x in affected_controls[i]],
                            )
                        )
                    st.session_state.events = new_events
                    st.success("Hendelser lagret.")
                except Exception as e:
                    st.error(f"Kunne ikke lagre hendelser: {e}")


# ---------------------- Tab: Resultater (historikk) ----------------------
with tab_results:
    st.subheader("Historikk")
    if not st.session_state.weekly_summaries:
        st.info("Ingen resultater ennå. Kjør en simulering i Dashboard.")
    else:
        # Samle alle uker i én tabell (lang format)
        all_rows = []
        for wk, rows in st.session_state.weekly_summaries:
            for (name, wk_chg, total_chg, price) in rows:
                all_rows.append(
                    {"Uke": wk, "Aksje": name, "Uke-endring (%)": wk_chg, "Total endring (%)": total_chg, "Ny pris (kr)": price}
                )
        big_df = pd.DataFrame(all_rows)
        st.dataframe(big_df, use_container_width=True, hide_index=True)

        csv_bytes = big_df.to_csv(index=False, sep=";").encode("utf-8")
        st.download_button("⬇️ Last ned alle resultater (CSV)", data=csv_bytes, file_name="aksjesimulering_resultater.csv", mime="text/csv")

        st.markdown("### Graf – siste uke")
        if st.session_state.last_fig_png:
            st.image(st.session_state.last_fig_png, use_container_width=True)
        else:
            st.caption("Ingen graf lagret enda.")


# ---------------------- Simuleringskjerne ----------------------
def simulate_one_week(rng: np.random.RandomState | np.random.RandomState) -> Tuple[bool, str]:
    """Simulerer én uke (5 dager / 30 tidssteg) og oppdaterer session_state.
    Returnerer (ok, msg)."""

    # Valider inputs
    if not st.session_state.stocks:
        return False, "Ingen aksjer definert."
    if not ensure_unique_names([s.name for s in st.session_state.stocks]):
        return False, "Aksjenavn må være unike."
    if not ensure_unique_names([e.name for e in st.session_state.events]) and st.session_state.events:
        return False, "Hendelsesnavn må være unike."

    # Første uke: init verdier
    if st.session_state.current_week == 0:
        st.session_state.initial_values = {s.name: float(s.value) for s in st.session_state.stocks}
        st.session_state.previous_values = {s.name: float(s.value) for s in st.session_state.stocks}
        st.session_state.histories = {s.name: [float(s.value)] for s in st.session_state.stocks}

    if st.session_state.current_week >= st.session_state.total_weeks:
        return False, "Alle uker er allerede kjørt."

    # Parametre
    total_steps = 30
    days = ["Mandag", "Tirsdag", "Onsdag", "Torsdag", "Fredag"]

    # Kopier “mutable” state for denne uken
    stocks_local = [Stock(**asdict(s)) for s in st.session_state.stocks]
    histories = {k: list(v) for k, v in st.session_state.histories.items()}
    prev_vals = dict(st.session_state.previous_values)
    init_vals = dict(st.session_state.initial_values)

    active_events: List[Tuple[Event, int]] = []  # (event, steps_left)
    new_events_this_week = 0
    max_events = int(st.session_state.max_events_per_week)

    # Event-logg for denne uken
    week_log: List[str] = []

    # Simuler tidssteg
    for step in range(total_steps):
        # Trigger nye hendelser (maks per uke)
        if st.session_state.events and new_events_this_week < max_events:
            for ev in st.session_state.events:
                roll = rng.randint(1, 101)
                if roll <= int(round(ev.probability)):
                    active_events.append((ev, 3))
                    new_events_this_week += 1
                    week_log.append(f"Uke {st.session_state.current_week+1}, steg {step+1}: {ev.name} (påvirker {', '.join(ev.affected_stocks) or 'ingen'})")
                    # en om gangen per steg
                    break

        # Prisoppdatering
        for s in stocks_local:
            total_change = rng.uniform(s.min_change, s.max_change)
            for (ev, _left) in active_events:
                if s.name in ev.affected_stocks:
                    total_change += ev.impact

            # Normalisering (samme idé som Tk-versjon)
            normalized = max(min(s.value, 300), 20)
            adjustment_factor = 100 / normalized
            adjusted_change = total_change * adjustment_factor

            s.value = max(s.value * (1 + adjusted_change / 100.0), 5.0)
            histories[s.name].append(float(s.value))

        # Tikk ned aktive hendelser
        new_active = []
        for (ev, left) in active_events:
            left -= 1
            if left > 0:
                new_active.append((ev, left))
        active_events = new_active

    # Uke ferdig – bygg figur
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Aksjeutvikling – uke {st.session_state.current_week+1}")
    ax.set_xlabel("Tid")
    ax.set_ylabel("Verdi")
    ax.set_xlim(0, total_steps)
    ax.set_xticks([i * 6 for i in range(len(days))])
    ax.set_xticklabels(days)
    for name, series in histories.items():
        ax.plot(range(len(series)), series, label=name)
    ax.legend(loc="best")
    fig_png = fig_to_png_bytes(fig)
    plt.close(fig)

    # Oppsummering
    rows = []
    for s in stocks_local:
        name = s.name
        new_price = float(s.value)
        last_price = float(prev_vals[name])
        start_price = float(init_vals[name])
        week_change = ((new_price - last_price) / last_price) * 100.0 if last_price else 0.0
        total_change = ((new_price - start_price) / start_price) * 100.0 if start_price else 0.0
        rows.append((name, round(week_change, 2), round(total_change, 2), round(new_price, 2)))

    # Commit uke → session_state
    st.session_state.current_week += 1
    st.session_state.histories = histories
    st.session_state.previous_values = {s.name: float(s.value) for s in stocks_local}
    # Ikke overskriv initial_values etter uke 1
    st.session_state.stocks = stocks_local
    st.session_state.last_fig_png = fig_png
    st.session_state.weekly_summaries.append((st.session_state.current_week, rows))
    st.session_state.event_log.extend(week_log)

    # Vis “toasts” for hendelser
    for msg in week_log:
        st.toast(msg, icon="🔔")

    return True, "OK"
