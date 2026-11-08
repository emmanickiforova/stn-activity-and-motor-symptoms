"""Групповые результаты и клинические корреляции субталамического ядра."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MEASURES = {"phase_rank": "Фазовая привязка", "lfp_beta": "Бета в LFP над фоном"}
OUTCOMES = {"contra_ar_off": "Брадикинезия и ригидность в OFF",
            "contra_ar_pct": "Улучшение на леводопе"}
TEAL, GRAY, INK = "#066e76", "#778896", "#233847"


def number(value, digits=2):
    return f"{value:.{digits}f}".replace("-", "−").replace(".", ",")


def style(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#b0bdc4")
    ax.tick_params(axis="both", length=0, pad=6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4eaed")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: number(x)))


def group_figure():
    df = pd.read_csv(RESULTS / "group.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4))
    fig.subplots_adjust(left=.15, right=.92, bottom=.20, top=.77)
    fig.text(.055, .91, "Фазовая привязка по частотным диапазонам", fontsize=17, weight="bold", color=INK)
    fig.text(.055, .83, "372 нейрона · 22 пациента · отрезки показывают 95%-е интервалы", color=GRAY)
    ax.axvline(.5, ls="--", lw=1, color=INK)
    for i, r in enumerate(df.itertuples()):
        color = TEAL if r.band == "Бета" else GRAY
        ax.hlines(i, r.ci_low, r.ci_high, color=color, lw=3)
        ax.scatter(r.mean_rank, i, color=color, s=40, zorder=3)
        ax.text(.653, i, number(r.mean_rank, 3), va="center", ha="left", color=color)
    ax.set(yticks=range(len(df)), yticklabels=df.band, ylim=(3.55, -.55), xlim=(.35, .675),
           xticks=[.35,.4,.45,.5,.55,.6,.65], xlabel="Средний нормированный ранг PPC · случайный уровень 0,50")
    style(ax)
    fig.savefig(RESULTS / "group_rank.png", dpi=160)
    plt.close(fig)


def clinic_figure():
    rows = pd.read_csv(RESULTS / "clinic.csv").set_index(["measure", "outcome"])
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 5.3))
    fig.subplots_adjust(left=.06, right=.96, top=.73, bottom=.15, hspace=.80, wspace=.18)
    fig.text(.05,.92,"Связь бета-активности с двигательными симптомами",weight="bold",fontsize=16,color=INK)
    fig.text(.05,.85,"Противоположная сторона тела · коэффициент Спирмена и 95%-й бутстреп-интервал",color=GRAY)
    for i, (x, name) in enumerate(MEASURES.items()):
        for j, (y, outcome) in enumerate(OUTCOMES.items()):
            ax = axes[i,j]
            r = rows.loc[(x,y)]
            ax.set_title(outcome if i == 0 else "", fontsize=11, pad=12, color=INK)
            ax.text(.02,.93,name, transform=ax.transAxes, fontsize=10, color=INK, weight="bold")
            ax.text(.02,.05,f"ρ = {number(r.rho)}     p = {number(r.p,3)}     {int(r.hemispheres)} полуш. / {int(r.patients)} пац.",transform=ax.transAxes, fontsize=9)
            ax.set(xlim=(-.65,.65),ylim=(0,1),yticks=[],xticks=[-.6,-.3,0,.3,.6])
            ax.axvline(0,color=GRAY,ls="--",lw=1)
            ax.hlines(.5,r.ci_low,r.ci_high,color=TEAL,lw=3)
            ax.scatter(r.rho,.5,color=TEAL,s=50,zorder=3)
            style(ax)
    fig.text(.5,.035,"Коэффициент корреляции Спирмена",ha="center")
    fig.savefig(RESULTS / "phase_and_clinic.png", dpi=160)
    plt.close(fig)


def hemisphere_scatter(data, destination):
    """График и четыре коэффициента по таблице с одной строкой на полушарие."""
    if data.duplicated(["patient", "hemisphere"]).any():
        raise ValueError("Ожидается одна строка на полушарие пациента.")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    stats = []
    for i, (x, name) in enumerate(MEASURES.items()):
        for j, (y, label) in enumerate(OUTCOMES.items()):
            ax = axes[i,j]
            d = data[["patient", x, y]].dropna()
            rho = spearmanr(d[x], d[y]).statistic
            stats.append((x, y, len(d), d.patient.nunique(), rho))
            for _, pair in d.groupby("patient"):
                if len(pair) == 2:
                    ax.plot(pair[x], pair[y], color="#c9d3d7", lw=.9, zorder=1)
            ax.scatter(d[x], d[y], color=TEAL, s=25, zorder=2)
            ax.set(xlabel=name, ylabel="Баллы" if j == 0 else "Доля улучшения",
                   title=f"{label}\nρ = {number(rho)}, n = {len(d)}")
            style(ax)
    fig.savefig(destination, dpi=160)
    plt.close(fig)
    return pd.DataFrame(stats, columns=["measure", "outcome", "hemispheres", "patients", "rho"])


if __name__ == "__main__":
    group_figure()
    clinic_figure()
