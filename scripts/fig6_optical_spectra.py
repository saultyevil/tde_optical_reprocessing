from platform import system
import pypython
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from pypython import constants, spectrum

root = "tde_opt_spec"
m_bh = "3e6"
sm = 10
lw = 1.7
al = 0.75
inclination = "60"

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

important_lines = [
    [r"He \textsc{i}", 3889], [r"H$~\delta$", 4100], [r"H$~\gamma$", 4340], [r"He \textsc{ii}", 4686],
    [r"H$~\beta$", 4861], [r"He \textsc{i}", 5877], [r"H$~\alpha$", 6564], [r"He \textsc{i}", 7067]
]

xmin = 3900
xmax = 7250

fig, ax = plt.subplots(2, 2, figsize=(13.3, 10), sharey=True, sharex=True)
ax = ax.flatten()
fmt = ScalarFormatter()
fmt.set_scientific(False)

i = 0

for subgrid, names, meta in zip(grids, grid_names, meta_names):
    for j, (model, name) in enumerate(zip(subgrid, names)):

        s = spectrum.Spectrum(root, model, log_spec=False, smooth=sm, distance=100 * 1e6)
        x, y = pypython.get_xy_subset(s["Lambda"], s[inclination], xmin, xmax)
        ax[i].plot(x, y, label=name, alpha=al)

    ax[i].text(0.04, 0.12, meta, transform=ax[i].transAxes, fontsize=18)
    ax[i].legend(loc="lower left", ncol=3, fontsize=13)
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
    ax[i].set_xlim(xmin, xmax)
    ax[i].set_ylim(5e-18, 7e-16)
    ax[i] = spectrum.plot.add_line_ids(ax[i], important_lines, linestyle="none", offset=0)

    ax[i].xaxis.set_major_formatter(fmt)
    ax[i].xaxis.set_minor_locator(MultipleLocator(1000))
    ax[i].xaxis.set_minor_formatter(fmt)
    ax[i].xaxis.set_ticklabels(["4000", "4000", "5000", "6000", "7000"], minor=True)

    i += 1

ax[-1] = spectrum.plot.add_line_ids(ax[-1], important_lines, linestyle="none", offset=0)

fig.text(0.02,
         0.5,
         r"Flux density at 100 Mpc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]",
         ha="center",
         va="center",
         rotation="vertical")

fig.text(0.5, 0.02, r"Rest-frame wavelength [\AA]", ha="center", va="center")

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.subplots_adjust(wspace=0, hspace=0)

fig.savefig("../p_figures/figure6_optical_spectra.pdf", dpi=300)
plt.show()
