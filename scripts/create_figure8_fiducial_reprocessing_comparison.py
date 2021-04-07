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

fig, ax = plt.subplots(1, 2, figsize=(12, 6))

twin_ax = []

the_models = [
    os_root + "PySims/tde_optical/tests/uv_fiducial/classic",
    os_root + "PySims/tde_optical/model_grids/new_grid_cmf_spec/3e6/Vinf/0_3"
]

the_roots = [
    "tde_cv", "tde_opt_cmf_spec"
]

the_names = [
    "Parkinson et al. 2020", "This work"
]

for i, (model, root, name) in enumerate(zip(the_models, the_roots, the_names)):
    print(i, model, root)
    ax_t = ax[i].twinx()
    s = spectrum.Spectrum(root, model)
    s.smooth(sm)
    s_opt_depth = spectrum.Spectrum(root, model, False, "spec_tau")
    s_opt_depth_inclinations = list(s_opt_depth.inclinations)
    try:
        s_opt_depth_inclinations.remove("45")
    except ValueError:
        pass

    opt_colours = []
    for k in range(len(s_opt_depth_inclinations)):
        opt_colours.append("C" + str(k + 1))

    emitted = s["Lambda"] * s["Emitted"]
    created = s["Lambda"] * s["Created"]

    ax[i].plot(s["Freq."], emitted * SCALE_FACTOR, label="Emergent", alpha=alpha)
    ax[i].plot(s["Freq."], created * SCALE_FACTOR, label="Disc", alpha=alpha)

    for k, opt_inc in enumerate(s_opt_depth_inclinations):
        od = s_opt_depth[opt_inc]
        if np.count_nonzero(od) != len(od):
            # This is to avoid plotting inclination angles which have no
            # optical depth values
            continue
        label = r"$\tau_{i =" + opt_inc + r"^{\circ}}$"
        ax_t.plot(s_opt_depth["Freq."], od, label=label, color=opt_colours[k], alpha=alpha)
        ax_t.set_yticks([])
        ax_t.set_yticks([], minor=True)

    ax[i].set_zorder(ax_t.get_zorder() + 1)
    ax[i].patch.set_visible(False)
    ax[i].text(0.02, 0.025, name, va="center", transform=ax[i].transAxes, fontsize=15)
    ax[i].set_ylim(1e-1 * SCALE_FACTOR, 5e2 * SCALE_FACTOR)
    ax[i].set_xlim(np.min(s["Freq."]), np.max(s["Freq."]))
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
    ax_t.set_xscale("log")
    ax_t.set_yscale("log")
    ax_t.set_ylim(0.3, 3e8)
    ax_t = plotutil.ax_add_line_ids(ax_t, edges, "none", offset=0, ynorm=0.87, logx=True)
    twin_ax.append(ax_t)

ax[1].legend(loc="upper left", ncol=1)
ax[1].set_yticks([])
ax[1].set_yticks([], minor=True)
twin_ax[0].legend(loc="upper left", ncol=1)

twin_ax[0].set_yticks([])
twin_ax[0].set_yticks([], minor=True)
locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
twin_ax[1].yaxis.set_major_locator(locmaj)
for label in twin_ax[0].yaxis.get_ticklabels()[2::2]:
    label.set_visible(False)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(.1, .2, .3, .4, .5, .6, .7, .8, .9), numticks=12)
twin_ax[1].yaxis.set_minor_locator(locmin)
twin_ax[1].yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())


fig.text(0.5, 0.03, "Rest frame Frequency [Hz]", ha="center", va="center")
twin_ax[1].set_ylabel("Continuum Optical Depth")
ax[0].set_ylabel(r"Flux at 100 Mpc [erg s$^{-1}$ cm$^{-2}$]")

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.subplots_adjust(wspace=0)
fig.savefig("../paper_figures/figure8_fiducial_reprocessing_comparison.pdf", dpi=300)
# fig.savefig("../paper_figures/paper1_vs_paper2_reprocessing.png", dpi=300)
plt.close()
