import numpy as np
from pypython import spectrum
from matplotlib import pyplot as plt
from platform import system
from pypython.physics.constants import C, ANGSTROM, PARSEC
from pypython import plotutil
import matplotlib

DEFAULT_DISTANCE = 100 * PARSEC
SCALED_DISTANCE = 100 * 1e6 * PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE ** 2 / SCALED_DISTANCE ** 2

plotutil.set_default_rcparams()

lw = 2
alpha = 0.7
sm = 5
root = "tde_opt_cmf_spec"
if system() == "Darwin":
    os_root = "/Users/saultyevil/"
else:
    os_root = "/home/saultyevil/"
home = os_root + "/PySims/tde_optical/model_grids/new_grid_cmf_spec/"

grids = [
    ["3e6/Mdot_acc/0_05", "3e6/Mdot_acc/0_15", "3e6/Mdot_acc/0_5"],
    ["3e6/Mdot_wind/0_1", "3e6/Mdot_wind/0_3", "3e6/Mdot_wind/1_0"],
    ["3e6/Vinf/0_1", "3e6/Vinf/0_3", "3e6/Vinf/1_0"],
    ["1e6/Vinf/0_3", "3e6/Vinf/0_3", "1e7/Vinf/0_3"]
]

grid_names = [
    [
        r"$\dot{\mathrm{M}}_{\mathrm{disc}}$ = 0.05 $\dot{\mathrm{M}}_{\mathrm{Edd}}$",
        r"$\dot{\mathrm{M}}_{\mathrm{disc}}$ = 0.15 $\dot{\mathrm{M}}_{\mathrm{Edd}}$",
        r"$\dot{\mathrm{M}}_{\mathrm{disc}}$ = 0.50 $\dot{\mathrm{M}}_{\mathrm{Edd}}$"
    ],
    [
        r"$\dot{\mathrm{M}}_{\mathrm{wind}}$ = 0.1 $\dot{\mathrm{M}}_{\mathrm{disc}}$",
        r"$\dot{\mathrm{M}}_{\mathrm{wind}}$ = 0.3 $\dot{\mathrm{M}}_{\mathrm{disc}}$",
        r"$\dot{\mathrm{M}}_{\mathrm{wind}}$ = 1.0 $\dot{\mathrm{M}}_{\mathrm{disc}}$"
    ],
    [
        r"$v_{\infty}$ = 0.1 $v_{\mathrm{esc}}$",
        r"$v_{\infty}$ = 0.3 $v_{\mathrm{esc}}$",
        r"$v_{\infty}$ = 1.0 $v_{\mathrm{esc}}$"
    ],
    [
        r"Fiducial $\mathrm{M}_{\mathrm{BH}} = 10^{6}~\mathrm{M}_{\odot}$",
        r"Fiducial $\mathrm{M}_{\mathrm{BH}} = 3 \times 10^{6}~\mathrm{M}_{\odot}$",
        r"Fiducial $\mathrm{M}_{\mathrm{BH}} = 10^{7}~\mathrm{M}_{\odot}$"
    ]
]

edges = [
    [r"He \textsc{ii}", 229],
    [r"He \textsc{i}", 504],
    ["Lyman", 912],
    # ["Ca I", 2028],
    # ["Al I", 2071],
    # ["Balmer", 3646],
    # ["Paschen", 8204],
]

for i in range(len(edges)):
    edges[i][1] = C / (edges[i][1] * ANGSTROM)

# ##############################################################################
#
# ##############################################################################

fig, ax = plt.subplots(4, 3, figsize=(13, 15.6), sharex=True, sharey=True)
twin_ax = np.empty((4, 3), dtype=type(ax[0, 0]))

for i, (subgrid, subgrid_name) in enumerate(zip(grids, grid_names)):
    for j, (model, model_name) in enumerate(zip(subgrid, subgrid_name)):
        ax_t = ax[i, j].twinx()
        s = spectrum.Spectrum(root, home + model)
        s.smooth(sm)
        s_opt_depth = spectrum.Spectrum(root, home + model, False, "spec_tau")
        s_opt_depth_inclinations = s_opt_depth.inclinations
        opt_colours = []
        for k in range(len(s_opt_depth_inclinations)):
            opt_colours.append("C" + str(k + 1))

        emitted = s["Lambda"] * s["Emitted"]
        created = s["Lambda"] * s["Created"]
        ax[i, j].plot(s["Freq."], emitted * SCALE_FACTOR, label="Emergent", alpha=alpha)
        ax[i, j].plot(s["Freq."], created * SCALE_FACTOR, label="Disc", alpha=alpha)

        for k, opt_inc in enumerate(s_opt_depth_inclinations):
            od = s_opt_depth[opt_inc]
            if np.count_nonzero(od) != len(od):
                # This is to avoid plotting inclination angles which have no
                # optical depth values
                continue
            label = r"$\tau_{i =" + opt_inc + r"^{\circ}}$"
            ax_t.plot(s_opt_depth["Freq."], od, label=label, color=opt_colours[k], alpha=alpha)

        ax[i, j].text(0.03, 0.03, model_name, transform=ax[i, j].transAxes, fontsize=15)
        ax[i, j].set_xscale("log")
        ax[i, j].set_yscale("log")
        ax[i, j].set_xlim(np.min(s["Freq."]), np.max(s["Freq."]))
        ax[i, j].set_ylim(5e-2 * SCALE_FACTOR, 2e3 * SCALE_FACTOR)
        ax_t.set_xscale("log")
        ax_t.set_yscale("log")

        # Disable tick markers on RHS axes
        ax_t.set_yticks([])
        ax_t.set_yticks([], minor=True)

        if (i == 0 and j == 0) or (i == 0 and j == 1):
            ax_t = plotutil.ax_add_line_ids(ax_t, edges, "none", ynorm=0.88, logx=True)
        else:
            ax_t = plotutil.ax_add_line_ids(
                ax_t, plotutil.photoionization_edges(True), "none", ynorm=0.88, logx=True
            )

        ax[i, j].set_zorder(ax_t.get_zorder() + 1)
        ax[i, j].patch.set_visible(False)
        twin_ax[i, j] = ax_t

for this_twin_ax in twin_ax.flatten():
    this_twin_ax.set_ylim(5e-2, 7e8)

# Add pretty tick markers to RHS axes, needs to be done after the limits have
# been set, otherwise bad shit happens

for i in range(4):
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    twin_ax[i, -1].yaxis.set_major_locator(locmaj)
    for label in twin_ax[i, -1].yaxis.get_ticklabels()[2::2]:
        label.set_visible(False)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=12)
    twin_ax[i, -1].yaxis.set_minor_locator(locmin)
    twin_ax[i, -1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

ax[0, 1].legend(loc="upper left")
twin_ax[0, 0].legend(loc="upper left")

fig.text(0.5, 0.019, "Rest frame Frequency [Hz]", ha="center", va="center")
fig.text(0.985, 0.5, "Continuum Optical Depth", rotation="vertical", ha="center", va="center")
fig.text(
    0.015, 0.5, r"Flux at 100 Mpc [erg s$^{-1}$ cm$^{-2}$]", rotation="vertical", ha="center", va="center",
)

fig.tight_layout(rect=[0.015, 0.015, 0.985, 0.985])
fig.subplots_adjust(hspace=0, wspace=0)
fig.savefig("../paper_figures/figure7_model_reprocessing.pdf", dpi=300)
# fig.savefig("../paper_figures/model_reprocessing.png", dpi=300)
plt.close()
