"""Запуск из step11_clinic.ipynb после создания nd и H (см. README)."""
from pathlib import Path

from plot_results import hemisphere_scatter


def draw_from_step11(nd, H, destination):
    keys = ["folder", "hemisphere"]
    grouped = (nd.rename(columns={"patient": "folder"})
               .groupby(keys)["rank_бета_все"]
               .agg(phase_rank="mean", available_ranks="count", neurons="size")
               .reset_index())
    clinical = H[keys + ["n_neurons", "lfp_beta", "contra_ar_off", "contra_ar_pct"]]
    d = clinical.merge(grouped, on=keys, how="left", validate="one_to_one")
    if (d["neurons"].fillna(0) != d["n_neurons"].fillna(0)).any():
        raise ValueError("Число нейронов в nd и H различается. Запусти обе ячейки очистки и агрегации.")
    d.loc[d.available_ranks < 3, "phase_rank"] = float("nan")
    d = d.rename(columns={"folder": "patient"})
    return hemisphere_scatter(d, Path(destination))
