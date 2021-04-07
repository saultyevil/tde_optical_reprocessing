from matplotlib import pyplot as plt
import numpy as np
from pypython import spectrum
from pypython.util import smooth_array
from matplotlib.ticker import ScalarFormatter, MultipleLocator
from pypython.physics import constants
from pypython import plotutil
from pypython import util

DEFAULT_DISTANCE = 100 * constants.PARSEC
SCALED_DISTANCE = 100 * 1e6 * constants.PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE ** 2 / SCALED_DISTANCE ** 2

plotutil.set_default_rcparams()

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
    [r"Na \textsc{i}", 5891],
    [r"H$~\alpha$", 6564],
]

fmt = ScalarFormatter()
fmt.set_scientific(False)

new = spectrum.get_spectrum_files(cd="../tests/pcygni/rin_grid")
old = list(
    dict.fromkeys(spectrum.get_spectrum_files(this_root="tde_opt_cmf_spec", cd="../model_grids/new_grid_cmf_spec/3e6"))
)
model_new = "../tests/pcygni/rin_grid/Vinf/1_0/tde_opt_cmf.spec"
model_old = "../model_grids/new_grid_cmf_spec/3e6/Vinf/1_0/tde_opt_cmf_spec.spec"

root_new, cd_new = util.get_root_from_filepath(model_new)
root_old, cd_old = util.get_root_from_filepath(model_old)

s_new = spectrum.Spectrum(root_new, cd_new)
s_old = spectrum.Spectrum(root_old, cd_old)
s_new.smooth(sm)
s_old.smooth(sm)

fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
ax[0].plot(s_old["Lambda"], s_old["60"] * SCALE_FACTOR, alpha=alpha)
ax[0].plot(s_new["Lambda"], s_new["60"] * SCALE_FACTOR, alpha=alpha)

ax[1].plot(
    s_old["Lambda"], s_old["75"] * SCALE_FACTOR, label=r"$R_{\mathrm{co}} = R_{\mathrm{ISCO}}$",
    alpha=alpha
)
ax[1].plot(
    s_new["Lambda"], s_new["75"] * SCALE_FACTOR, label=r"$R_{\mathrm{co}} = 3.8~R_{\mathrm{ISCO}}$",
    alpha=alpha
)

for k in range(2):
    ax[k].set_xscale("log")
    ax[k].set_yscale("log")
    ax[k].set_xlim(1000, 7000)
    ax[k].set_ylim(3e-5 * SCALE_FACTOR, 2e-12)
    ax[k] = plotutil.ax_add_line_ids(
        ax[k], important_lines, linestyle="none", ynorm=0.84, logx=True, offset=0
    )

fig.text(
    0.017, 0.5, r"Flux Density at 100 Mpc [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", rotation="vertical", ha="center",
    va="center"
)

ax[1].set_xlabel(r"Rest frame Wavelength [\AA]")
ax[1].legend(loc="lower left",)
ax[0].text(0.95, 0.93, r"60$^{\circ}$", transform=ax[0].transAxes, fontsize=17)
ax[1].text(0.95, 0.93, r"75$^{\circ}$", transform=ax[1].transAxes, fontsize=17)

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.subplots_adjust(hspace=0)
fig.savefig("../paper_figures/figure11_Pcyg_model.pdf", dpi=300)
# fig.savefig("../paper_figures/pcygni_model_test.png", dpi=300)
plt.close()
