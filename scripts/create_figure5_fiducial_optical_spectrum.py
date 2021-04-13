from pypython import spectrum
from matplotlib import pyplot as plt
from pypython.physics import constants
from pypython import plotutil

DEFAULT_DISTANCE = 100 * constants.PARSEC
SCALED_DISTANCE = 100 * 1e6 * constants.PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE ** 2 / SCALED_DISTANCE ** 2

plotutil.set_default_rcparams()

root = "tde_opt"
m_bh = "3e6"
home = "../data/" + m_bh + "/"

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
# Actual fiducial model: optical spectrum
#
# ##############################################################################

xmin = 3750
xmax = 7250

fig, ax = plt.subplots(1, 1, figsize=(12, 5))

important_lines = [
    [r"Ly$\alpha$/N V", 1216],
    ["", 1240],
    ["O V/Si IV", 1371],
    ["", 1400],
    ["C IV", 1549],
    ["He II", 1640],
    ["N III]", 1750],
    ["C III]", 1908],
    ["Mg II", 2798],
    ["He I", 3889],
    [r"H$_{\delta}$", 4100],
    [r"H$_{\gamma}$", 4340],
    ["He II", 4641],
    [r"H$_{\beta}$", 4861],
    ["Na I", 5891],
    [r"H$_{\alpha}$", 6564],
    ["He II", 7067],
]

model = grids[2][1]

s = spectrum.Spectrum(root, home + model)
s.smooth(sm)
s_inclinations = s.inclinations
geo = ["out: ", "in : ", "in : ", "out: ", "out: "]

for inclination, inwind in zip(s_inclinations, geo):
    ax.plot(
        s["Lambda"], s[inclination] * SCALE_FACTOR, alpha=al, label=str(inclination) + r"$^{\circ}$"
    )

ax.set_xlim(xmin, xmax)
ax = plotutil.ax_add_line_ids(ax, important_lines, ynorm=0.88, linestyle="none", offset=0)
ax.set_ylim(1e-4 * SCALE_FACTOR, 8e-3 * SCALE_FACTOR)
ax.legend(loc="lower left", ncol=5)
ax.set_xscale("linear")
ax.set_yscale("log")
ax.set_ylabel("Flux Density at 100 Mpc\n" + r"[erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
ax.set_xlabel(r"Rest frame Wavelength [\AA]")

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
plt.show()
