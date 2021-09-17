from platform import system

import matplotlib
import numpy as np
import pypython
from matplotlib import pyplot as plt
from pypython import spectrum
from pypython.constants import ANGSTROM, PARSEC, C

root = "tde_opt_sed"
m_bh = "3e6"
sm = 10
lw = 1.7
alpha = 0.75

if system() == "Darwin":
    home = "/Users/saultyevil/"
else:
    home = "/home/saultyevil/"

grids = [
    ["Mdot_acc/0_05", "Mdot_acc/0_15", "Mdot_acc/0_5"], ["Mdot_wind/0_1", "Mdot_wind/0_3", "Mdot_wind/1_0"],
    ["Vinf/0_1", "Vinf/0_3", "Vinf/1_0"], ["Mdot_acc/0_15", "Mdot_acc/0_15", "Mdot_acc/0_15"]
]

for i in range(3):
    for j in range(3):
        grids[i][j] = home + "PySims/tde_optical/p_response/12_grid_final/3e6/" + grids[i][j]

for i, mbh in enumerate(["1e6", "3e6", "1e7"]):
    grids[3][i] = home + "PySims/tde_optical/p_response/12_grid_final/" + mbh + "/" + grids[3][i]

print(grids)

grid_names = [
    [
        r"0.05 $\dot{\mathrm{M}}_{\mathrm{Edd}}$", r"0.15 $\dot{\mathrm{M}}_{\mathrm{Edd}}$",
        r"0.50 $\dot{\mathrm{M}}_{\mathrm{Edd}}$"
    ],
    [   r"0.1 $\dot{\mathrm{M}}_{\mathrm{disc}}$", r"0.3 $\dot{\mathrm{M}}_{\mathrm{disc}}$",
        r"1.0 $\dot{\mathrm{M}}_{\mathrm{disc}}$"
    ], 
    [
        r"0.1 $v_{\mathrm{esc}}$", r"0.3 $v_{\mathrm{esc}}$", r"1.0 $v_{\mathrm{esc}}$"
    ],
    [
        r"$10^{6}~\rm M_{\odot}$", r"$3 \times 10^{6}~\rm M_{\odot}$", r"$10^{7}~\rm M_{\odot}$"
    ]
]

meta_names = [
    r"$\dot{\mathrm{M}}_{\mathrm{disc}}$", r"$\dot{\mathrm{M}}_{\mathrm{wind}}$", r"$v_{\infty}$", 
    r"$\text{M}_{\rm BH}$"
]

edges = [
    [r"He \textsc{ii}", 229],
    [r"He \textsc{i}", 504],
    ["Lyman", 912],
]

for i in range(len(edges)):
    edges[i][1] = C / (edges[i][1] * ANGSTROM)

# ##############################################################################
#
# ##############################################################################

fig, ax = plt.subplots(4, 3, figsize=(13, 15.6), sharex=True, sharey=False)
twin_ax = np.empty((4, 3), dtype=type(ax[0, 0]))

for i, (subgrid, subgrid_name) in enumerate(zip(grids, grid_names)):
    for j, (model, model_name) in enumerate(zip(subgrid, subgrid_name)):

        ax_t_opt = ax[i, j].twinx()
        spec = pypython.Spectrum(root, model, smooth=sm, distance=100 * 1e6)
        spec.convert_flux_to_luminosity()
        spec_tau = pypython.Spectrum("tde_opt_cmf_spec_opt", model.replace("12_grid_final", "11_grid_lambda_2"), default="spec_tau")

        opt_colours = []
        for k in range(len(spec_tau["spec_tau"].inclinations)):
            opt_colours.append("C" + str(k + 1))

        emitted = spec["Lambda"] * spec["Emitted"]
        created = spec["Lambda"] * spec["Created"]
        ax[i, j].plot(spec["Freq."], emitted, label="Emergent", alpha=alpha)
        ax[i, j].plot(spec["Freq."], created, label="Disc", alpha=alpha)

        for k, opt_inc in enumerate(spec_tau["spec_tau"].inclinations):
            optical_depth_inclination = spec_tau["spec_tau"][opt_inc]
            if np.count_nonzero(optical_depth_inclination) != len(optical_depth_inclination):
                continue
            label = r"$\tau_{i =" + opt_inc + r"^{\circ}}$"
            ax_t_opt.plot(spec_tau["spec_tau"]["Freq."],
                      optical_depth_inclination,
                      label=label,
                      color=opt_colours[k],
                      alpha=alpha)

        ax[i, j].text(0.95, 0.03, model_name, transform=ax[i, j].transAxes, fontsize=15, ha="right")
        ax[i, j].set_xscale("log")
        ax[i, j].set_yscale("log")
        ax[i, j].set_xlim(np.min(spec["spec_tau"]["Freq."]), np.max(spec["spec_tau"]["Freq."]))    
        ax[i, j].set_ylim(2e40, 2e45)
        
        ax_t_opt.set_xscale("log")
        ax_t_opt.set_yscale("log")

        # Disable tick markers on RHS axes

        ax_t_opt.set_yticks([])
        ax_t_opt.set_yticks([], minor=True)

        if (i == 0 and j == 0) or (i == 0 and j == 1):
            ax_t_opt = spectrum.plot.add_line_ids(ax_t_opt, edges, "none", ynorm=0.88, offset=0)
        else:
            ax_t_opt = spectrum.plot.add_line_ids(ax_t_opt,
                                              spectrum.plot.photoionization_edges(units=spectrum.SpectrumUnits.f_nu),
                                              "none",
                                              ynorm=0.88,
                                              offset=0)

        ax[i, j].set_zorder(ax_t_opt.get_zorder() + 1)
        ax[i, j].patch.set_visible(False)

        # Store the twin ax in the twin_ax array
        ax_t_opt.set_ylim(2e-2, 7e9)
        twin_ax[i, j] = ax_t_opt

# Add pretty tick markers to RHS axes, needs to be done after the limits have
# been set, otherwise bad shit happens

for i in range(4):
    for j in range(2):
        ax[i, j + 1].set_yticks([])
        ax[i, j + 1].set_yticks([], minor=True)


# n_rows = 4
# for i in range(n_rows):
#     locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
#     ax[i, 0].yaxis.set_major_locator(locmaj)

#     for label in ax[i, 0].yaxis.get_ticklabels()[2::2]:
#         label.set_visible(False)

#     locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=12)
#     ax[i, 0].yaxis.set_minor_locator(locmin)
#     ax[i, 0].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


n_rows = 4
for i in range(n_rows):
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    twin_ax[i, -1].yaxis.set_major_locator(locmaj)

    for label in twin_ax[i, -1].yaxis.get_ticklabels()[2::2]:
        label.set_visible(False)

    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=12)
    twin_ax[i, -1].yaxis.set_minor_locator(locmin)
    twin_ax[i, -1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# rest of clean up fuck sake

ax[0, 1].legend(loc="upper left")
twin_ax[0, 0].legend(loc="upper left")

fig.text(0.5, 0.019, "Rest-frame frequency [Hz]", ha="center", va="center")
fig.text(0.985, 0.5, "Continuum optical depth", rotation="vertical", ha="center", va="center")
fig.text(0.015, 0.5, r"Flux [erg s$^{-1}$ cm$^{-2}$]", rotation="vertical", ha="center", va="center")

fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
fig.subplots_adjust(hspace=0, wspace=0)

fig.savefig("../p_figures/figure7_model_reprocessing.pdf", dpi=300)

plt.show()
