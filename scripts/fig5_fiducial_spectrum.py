from platform import system

import pypython
from matplotlib import pyplot as plt

DEFAULT_DISTANCE = 100 * pypython.constants.PARSEC
SCALED_DISTANCE = 100 * 1e6 * pypython.constants.PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE**2 / SCALED_DISTANCE**2

m_bh = "3e6"
root = "tde_opt_spec"
sm = 10
lw = 1.7
al = 0.75
inclination = "60"

if system() == "Darwin":
    home = "/Users/saultyevil/"
else:
    home = "/home/saultyevil/"

home += "PySims/tde_optical/p_response/12_grid_final/" + m_bh + "/"

important_lines = [[r"He \textsc{i}", 3889], [r"H$~\delta$", 4100], [r"H$~\gamma$", 4340], [r"He \textsc{ii}", 4686],
                   [r"H$~\beta$", 4861], [r"He \textsc{i}", 5877], [r"H$~\alpha$", 6564], [r"He \textsc{i}", 7067]]

# ##############################################################################
#
# Actual fiducial model: optical spectrum
#
# ##############################################################################

xmin = 4000
xmax = 7250

s = pypython.Spectrum(root, home + "Mdot_acc/0_15", log_spec=False, smooth=sm, distance=100 * 1e6)

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

for inclination in s.inclinations:
    x, y = pypython.get_xy_subset(s["Lambda"], s[inclination], xmin, xmax)
    ax.plot(x, y, alpha=al, label=str(inclination) + r"$^{\circ}$")

ax.set_xlim(xmin, xmax)
# ax.set_ylim(1e-5 * SCALE_FACTOR, 3e-3 * SCALE_FACTOR)
ax.legend(loc="lower left", ncol=5)
ax.set_ylabel("Flux density at 100 Mpc\n" + r"[erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
ax.set_xlabel(r"Rest-frame wavelength [\AA]")
ax = pypython.plot.set_axes_scales(ax, "logy")
ax = pypython.spectrum.plot.add_line_ids(ax, important_lines, ynorm=0.92, linestyle="none", offset=0)

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.savefig("../p_figures/figure5_fiducial_optical_spectrum.pdf", dpi=300)

plt.show()
