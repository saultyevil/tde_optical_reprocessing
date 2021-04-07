from pypython import spectrum
from matplotlib import pyplot as plt
from platform import system
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from pypython.physics import constants
from pypython import plotutil

DEFAULT_DISTANCE = 100 * constants.PARSEC
SCALED_DISTANCE = 100 * 1e6 * constants.PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE ** 2 / SCALED_DISTANCE ** 2

plotutil.set_default_rcparams()

root = "tde_opt_cmf_spec"
m_bh = "3e6"
if system() == "Darwin":
    home = "/Users/saultyevil/"
else:
    home = "/home/saultyevil/"
home += "/PySims/tde_optical/model_grids/new_grid_cmf_spec/" + m_bh + "/"

grids = [
    ["Mdot_acc/0_05", "Mdot_acc/0_15", "Mdot_acc/0_5"],
    ["Mdot_wind/0_1", "Mdot_wind/0_3", "Mdot_wind/1_0"],
    ["Vinf/0_1", "Vinf/0_3", "Vinf/1_0"]
]

grid_names = [
    [
        r"0.05 $\dot{\mathrm{M}}_{\mathrm{Edd}}$",
        r"0.15 $\dot{\mathrm{M}}_{\mathrm{Edd}}$",
        r"0.50 $\dot{\mathrm{M}}_{\mathrm{Edd}}$"
    ],
    [
        r"0.1 $\dot{\mathrm{M}}_{\mathrm{disc}}$",
        r"0.3 $\dot{\mathrm{M}}_{\mathrm{disc}}$",
        r"1.0 $\dot{\mathrm{M}}_{\mathrm{disc}}$"
    ],
    [
        r"0.1 $v_{\mathrm{esc}}$",
        r"0.3 $v_{\mathrm{esc}}$",
        r"1.0 $v_{\mathrm{esc}}$"
    ]
]

meta_names = [
    r"$\dot{\mathrm{M}}_{\mathrm{disc}}$",
    r"$\dot{\mathrm{M}}_{\mathrm{wind}}$",
    r"$v_{\infty}$"
]

sm = 1
lw = 1.7
al = 0.6
inclination = "60"

important_lines = [
    [r"Ly$~\alpha$/N \textsc{v}", 1216],
    ["", 1240],
    [r"O \textsc{v}/Si \textsc{iv}", 1371],
    ["", 1400],
    [r"C \textsc{iv}", 1549],
    [r"He \textsc{ii}", 1640],
    [r"N \textsc{iii]}", 1750],
    [r"C \textsc{iii]}", 1908],
    [r"Mg \textsc{ii}", 2798],
    [r"He \textsc{i}", 3889],
    [r"H$~\delta$", 4100],
    [r"H$~\gamma$", 4340],
    [r"He \textsc{ii}", 4686],
    [r"H$~\beta$", 4861],
    [r"Na \textsc{i}", 5891],
    [r"H$~\alpha$", 6564],
]

# ##############################################################################
#
# Optical spectra
#
# ##############################################################################

xmin = 3750
xmax = 7250

fig, ax = plt.subplots(2, 2, figsize=(13.3, 10), sharey=True, sharex=True)
ax = ax.flatten()

root = "tde_opt_cmf_spec"
fiducial = "/Vinf/0_3/"
if system() == "Darwin":
    home_bh = "/Users/saultyevil/"
else:
    home_bh = "/home/saultyevil/"
home_bh += "/PySims/tde_optical/model_grids/new_grid_cmf_spec/"
models_bh = [
    home_bh + "1e6" + fiducial,
    home_bh + "3e6" + fiducial,
    home_bh + "1e7" + fiducial
]
names_bh = [
    r"$10^{6}~\rm M_{\odot}$",
    r"$3 \times 10^{6}~\rm M_{\odot}$",
    r"$10^{7}~\rm M_{\odot}$"
]

for model, name in zip(models_bh, names_bh):
    s = spectrum.Spectrum(root, model)
    ax[-1].plot(s["Lambda"], s[inclination] * SCALE_FACTOR, label=name, alpha=al)
ax[-1].legend(loc="lower left", ncol=3)
ax[-1].text(0.04, 0.12, r"$\rm M_{\rm BH}$", transform=ax[-1].transAxes, fontsize=18)

fmt = ScalarFormatter()
fmt.set_scientific(False)

i = 0
for subgrid, names, meta in zip(grids, grid_names, meta_names):

    print(subgrid)
    for j, (model, name) in enumerate(zip(subgrid, names)):
        print(home + model)
        s = spectrum.Spectrum(root, home + model)
        s.smooth(sm)
        ax[i].plot(s["Lambda"], s[inclination] * SCALE_FACTOR, label=name, alpha=al)

    ax[i].text(0.04, 0.12, meta, transform=ax[i].transAxes, fontsize=18)
    ax[i].legend(loc="lower left", ncol=3, fontsize=13)
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(3e-5 * SCALE_FACTOR, 8e-3 * SCALE_FACTOR)
    ax[i] = plotutil.ax_add_line_ids(ax[i], important_lines, logx=True, linestyle="none", offset=0)

    ax[i].xaxis.set_major_formatter(fmt)
    ax[i].xaxis.set_minor_locator(MultipleLocator(1000))
    ax[i].xaxis.set_minor_formatter(fmt)
    ax[i].xaxis.set_ticklabels(["4000", "4000", "5000", "6000", "7000"], minor=True)

    i += 1

ax[-1] = plotutil.ax_add_line_ids(ax[-1], important_lines, logx=True, linestyle="none", offset=0)
fig.text(
    0.02, 0.5, r"Flux Density at 100 Mpc [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", ha="center", va="center",
    rotation="vertical"
)
fig.text(0.5, 0.02, r"Rest frame Wavelength [\AA]", ha="center", va="center")
fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.subplots_adjust(wspace=0, hspace=0)

fig.savefig("../paper_figures/figure6_optical_spectra.pdf", dpi=300)
# fig.savefig("../paper_figures/model_optical_spectra_i60.png", dpi=300)
plt.close()
