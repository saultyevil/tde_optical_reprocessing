#!/usr/bin/env python3

"""This is a docstring at the top of the script to stop the linter complaining."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from pypython.physics.constants import C, MSOL_PER_YEAR, MSOL, G, PI, MPROT, THOMPSON
from pypython.physics.accretiondisc import eddington_accretion_limit
from pypython import plotutil

plotutil.set_default_rcparams()

"""
The header for the file to be read in, is as follows:
CandidateName            Lobs1e44    Tobs1e4  MbhBulge1e6      Mbh1e6    MstarMsol    t0days  reference
"""


def estimate_accretion_rate(luminosity: float, efficiency: float):
    """
    Calculate a rough estimate for the mass accretion rate for a flat
    accretion disc given the luminosity and accretion efficiency.
    This function implements L = efficiency * mdot * c * c.
    """
    return luminosity / efficiency / C ** 2 / MSOL_PER_YEAR


def eddington_luminosity(Mbh: float):
    """Calculate the Eddington luminosity"""
    Mbh *= MSOL
    return 4 * PI * G * Mbh * C * MPROT / THOMPSON


def mscatter(x, y, ax=None, m=None, **kw):
    """Something about being able to plot seperate markers"""
    import matplotlib.markers as mmarkers
    if not ax:
        ax = plt.gca()
    sc = ax.scatter(x, y, **kw, zorder=3)
    if (m is not None) and (len(m) == len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


# Set the accretion efficiency and read in the data

accretion_efficiency = 0.1
data = pd.read_csv("../eddington_fractions/masses.txt", delim_whitespace=True)

# Now we can calculate a rough estimate for the accretion rate of the system
# assuming the disc is the dominant source of luminosity

data["MdotMsolyr"] = estimate_accretion_rate(data["Lobs1e44"] * 1e44, accretion_efficiency)
data["MeddMsolyr"] = eddington_accretion_limit(data["Mbh1e6"] * 1e6, accretion_efficiency) / MSOL_PER_YEAR
data["Fedd"] = data["MdotMsolyr"] / data["MeddMsolyr"]

# Create a plot of L / L_edd vs. Mbh

fig, ax = plt.subplots(figsize=(6.4, 5))
ax.set_xscale("log")
ax.set_yscale("log")

markers = []
the_labels = []

# Plot the objects of interest

kk = 0
ms = 10

obj_names = [
    "ASASSN-14li 52d",  # Mockler
    "AT2019qiz 49d"     # Ryu
]
obj_bh = np.array([
    [1.5],
    [9]
])
obj_ledd = np.array([
    [0.002747501253374],
    [0.018861979096545935]
])
obj_mbh_err = np.array([
    [0.1, 3],
    [0.1, 2]
]) * 1e6
obj_ledd_err = np.array([
    [0.3706415962150284, 0.04605170185988093],
    [0.3706415962150284, 0.04605170185988093]
])
symbols = [
    "v", "D"
]

for i in range(len(obj_names)):
    ax.plot(
        obj_bh[i] * 1e6, obj_ledd[i], symbols[i], label=obj_names[i], markersize=ms, zorder=5
    )

errb = plt.errorbar(
    obj_bh * 1e6, obj_ledd, xerr=obj_mbh_err, yerr=obj_ledd_err, fmt="none", zorder=0, color="k",
    alpha=0.3, elinewidth=0.8, capsize=1
)
legend = ax.legend(loc="upper right", fontsize=12)

# for index, row in data.iterrows():
#     d = row
#     if d["CandidateName"] in candidates:
#         markers.append(candidates_dict[d["CandidateName"]])
#         the_labels.append(d["CandidateName"])
#         ax.plot(
#             d["Mbh1e6"] * 1e6, d["Fedd"], candidates_dict[d["CandidateName"]], label=d["CandidateName"],
#             markersize=ms, zorder=2
#         )
#         kk += 1
#     else:
#         markers.append("o")
#         the_labels.append("_nolabel_")

# Plot the entire sample using the mscatter function

d = data
x_err = np.zeros((2, len(d)))
y_err = np.zeros_like(x_err)


for i, row in d.iterrows():
    x_err[0, i] = abs(row["MErrLower"])
    x_err[1, i] = row["MErrUpper"]

    y_err[0, i] = abs(row["LErrLower"])
    y_err[1, i] = row["LErrUpper"]

x_err *= 1e6
y_err *= 1e44 / eddington_luminosity(d["Mbh1e6"] * 1e6)

ratio = d["Lobs1e44"] * 1e44 / eddington_luminosity(d["Mbh1e6"] * 1e6)
sc = mscatter(data["Mbh1e6"] * 1e6, ratio, ax=ax, m=markers, c=np.log10(data["t0days"]))
errb = plt.errorbar(
    data["Mbh1e6"] * 1e6, ratio, xerr=x_err, yerr=y_err, fmt="none", zorder=0, color="k", alpha=0.3,
    elinewidth=0.8, capsize=1
)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(r"$\log_{10}(p)$ [days]")

# Add approximate lines for accretion (vertical axis) and Mbh (horizontal axis) parameter space

col = "k"
alph = 0.3
ax.axvline(1e6, 0, 1, linestyle="--", alpha=alph, color=col, zorder=0)
ax.axvline(1e7, 0, 1, linestyle="--", alpha=alph, color=col, zorder=0)

# Clean up plot for the paper

ax.set_xlabel(r"Black Hole Mass [$\rm M_{\odot}$]")
ax.set_ylabel(r"$L~/~L_{\rm Edd}$")
fig.tight_layout(rect=[0.00, 0.00, 0.99, 0.99])

plt.savefig("../paper_figures/figure3_Mbh_parameters.pdf", dpi=300)
# plt.savefig("../paper_figures/mbh_ryu2020_parameter_space.png", dpi=300)
plt.show()
