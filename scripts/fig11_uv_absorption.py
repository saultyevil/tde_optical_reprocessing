import numpy as np
import pypython
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from pypython import constants, spectrum, util

DEFAULT_DISTANCE = 100 * constants.PARSEC
SCALED_DISTANCE = 100 * 1e6 * constants.PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE**2 / SCALED_DISTANCE**2

sm = 1
alpha = 0.75
lw = 2

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
    [r"He \textsc{i}", 5877],
    [r"H$~\alpha$", 6564],
    [r"He \textsc{i}", 7067]
]

fmt = ScalarFormatter()
fmt.set_scientific(False)

model_new = "../etc/p_cygni/tde_opt_spec.spec"
model_old = "../3e6/Vinf/0_3/tde_opt_sed.spec"

root_new, cd_new = pypython.get_root_name(model_new)
root_old, cd_old = pypython.get_root_name(model_old)

s_new = spectrum.Spectrum(root_new, cd_new, log_spec=False)
s_old = spectrum.Spectrum(root_old, cd_old, log_spec=False)
s_new.smooth(sm)
s_old.smooth(sm)

fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
ax[0].plot(s_old["Lambda"], s_old["60"] * SCALE_FACTOR, alpha=alpha)
ax[0].plot(s_new["Lambda"], s_new["60"] * SCALE_FACTOR, alpha=alpha)

ax[1].plot(s_old["Lambda"], s_old["75"] * SCALE_FACTOR, label=r"$R_{\mathrm{co}} = R_{\mathrm{ISCO}}$", alpha=alpha)
ax[1].plot(s_new["Lambda"], s_new["75"] * SCALE_FACTOR, label=r"$R_{\mathrm{co}} = 3.8~R_{\mathrm{ISCO}}$", alpha=alpha)

ax[0].set_ylim(2e-17, 9e-13)
ax[1].set_ylim(6e-18, 9e-13)

for k in range(2):
    ax[k].set_xscale("log")
    ax[k].set_yscale("log")
    ax[k].set_xlim(1100, 7000)
    ax[k] = spectrum.plot.add_line_ids(ax[k], important_lines, linestyle="none", ynorm=0.84, offset=0)

fig.text(0.017,
         0.5,
         r"Flux density at 100 Mpc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]",
         rotation="vertical",
         ha="center",
         va="center")

ax[1].set_xlabel(r"Rest-frame wavelength [\AA]")
ax[1].legend(loc="lower left", )
ax[0].text(0.95, 0.93, r"60$^{\circ}$", transform=ax[0].transAxes, fontsize=17)
ax[1].text(0.95, 0.93, r"75$^{\circ}$", transform=ax[1].transAxes, fontsize=17)

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.subplots_adjust(hspace=0)
fig.savefig("../p_figures/figure11_Pcyg_model.pdf", dpi=300)
# fig.savefig("../paper_figures/pcygni_model_test.png", dpi=300)
plt.show()
