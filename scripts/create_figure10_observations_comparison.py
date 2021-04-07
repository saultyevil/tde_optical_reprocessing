from pypython import spectrum
from pypython import plotutil
from pypython.util import get_array_index, smooth_array
from pypython.physics import constants
import numpy as np
from matplotlib import pyplot as plt
from platform import system
import astropy.units as u
from dust_extinction.parameter_averages import F99, CCM89

plotutil.set_default_rcparams()

DEFAULT_DISTANCE = 100 * constants.PARSEC

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
    # [r"He \textsc{i}", 3889],
    [r"H$~\delta$", 4100],
    [r"H$~\gamma$", 4340],
    [r"He \textsc{ii}", 4686],
    [r"H$~\beta$", 4861],
    [r"Na \textsc{i}", 5891],
    [r"H$~\alpha$", 6564],
]

root = "tde_opt_cmf_spec"
if system() == "Darwin":
    home = "/Users/saultyevil/"
else:
    home = "/home/saultyevil/"
home += "PySims/tde_optical/model_grids/new_grid_cmf_spec/"
mymodels = [
    home + "1e6/Vinf/0_3",  # 1_0 looks kind of okay as well.
    # home + "3e6/Mdot_acc/0_5",
    home + "3e6/Vinf/1_0"   # 1e6 or 3e6/Mdot_acc/0_5 (75), 3e6/Vinf/1_0 (35) looks okay too
]                           # show both models, actually.
mylabels = [
    r"$10^{6}~\mathrm{M}_{\odot}$ fiducial model",
    # r"Fiducial $10^{6}~\mathrm{M}_{\odot}$ with $\dot{\rm M}_{\rm acc} = 0.5~\dot{\rm M}_{\rm Edd}$",
    r"$3 \times 10^{6}~\mathrm{M}_{\odot}$ fiducial model"
]
obs_home = "../observed_spectra/optical/"
obs_spectra = [
    obs_home + "ASASSN14li/ASASSN-14li_2015-01-03_Magellan-Baade---IMACS.ascii",  # UV emission
    # obs_home + "AT2019qiz/2019qiz_2019-11-26_ESO-NTT_EFOSC2-NTT_None.txt",        # UV absorption
    obs_home + "AT2019qiz/2019qiz_2019-11-26_ESO-NTT_EFOSC2-NTT_None.txt"         # UV absorption
]
obslabels = [
    "ASASSN-14li+54d",
    # None
    "AT2019qiz+49d",
]
redshifts = [
    0.0206,
    # 0.0151,
    0.0151
]
distances = [
    92.6,
    # 65.6,
    65.6
]
inclinations = [
    "10",
    # "75",
    "35"
]
Rv = [
    3.1,
    # 3.1,
    3.1
]
Ev = [
    0.07 / 3.1,
    # 0.0939,
    0.0939
]

sm = 1
alpha = 0.75
alpha2 = 0.75
linewidth = 2
xmin = 4000
xmax = 7000
b_min = 3000
b_max = 4600


def deredden(spectrum, Rv, Ebv, curve="CCM89"):
    """Deredden an input spectrum given the selective extinction and colour
    excess using the extinction curve of choice.
    Parameters
    ----------
    spec: np.ndarray
        The spectrum to de-redden. Must include wavelength and flux."""

    wl = spectrum[:, 0] * u.angstrom
    spectrum[:, 1] *= u.erg / u.s / u.cm / u.cm / u.AA

    if curve == "CCM89":
        curve = CCM89(Rv=Rv)
    elif curve == "F99":
        curve = F99(Rv=Rv)
    else:
        print("Unknown extinction curve {curve}. Returning original spectrum.")
        return spectrum

    spectrum[:, 1] /= curve.extinguish(wl, Ebv=Ebv)

    return spectrum


def add_to_ax(ax, smodel, stde, inc, distance, z, tdename, modelname):
    """Add the model and data to the provided ax between xmin and xmax"""

    print(tdename)
    scale_factor = DEFAULT_DISTANCE ** 2 / (distance * 1e6 * constants.PARSEC) ** 2

    stde /= z + 1
    if tdename is not None:
        itde1 = get_array_index(stde[:, 0], xmin)
        itde2 = get_array_index(stde[:, 0], xmax)
        ax.semilogy(
            stde[itde1:itde2, 0], smooth_array(stde[itde1:itde2, 1], sm), label=tdename, alpha=alpha
        )

    imodel1 = get_array_index(smodel["Lambda"], xmin)
    imodel2 = get_array_index(smodel["Lambda"], xmax)
    ax.semilogy(
        smodel["Lambda"][imodel1:imodel2], smooth_array(smodel[inc][imodel1:imodel2] * scale_factor, sm),
        label=modelname, alpha=alpha2
    )

    ax.text(0.98, 0.9, inc + r"$^{\circ}$", transform=ax.transAxes, fontsize=17, ha="right")
    ax.text(0.98, 0.84, "{} Mpc".format(distance), transform=ax.transAxes, fontsize=17, ha="right")

    return ax


fig, ax = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

n = 0
ax[0] = add_to_ax(
    ax[0], spectrum.Spectrum(root, mymodels[n]), deredden(np.loadtxt(obs_spectra[n]), Rv[n], Ev[n]), inclinations[n],
    distances[n], redshifts[n], obslabels[n], mylabels[n]
)
n += 1
ax[1] = add_to_ax(
    ax[1], spectrum.Spectrum(root, mymodels[n]), deredden(np.loadtxt(obs_spectra[n]), Rv[n], Ev[n]), inclinations[n],
    distances[n], redshifts[n], obslabels[n], mylabels[n]
)

n += 1

for n in range(2):
    ax[n].legend(loc="lower left")
    ax[n] = plotutil.ax_add_line_ids(ax[n], important_lines, linestyle="none", offset=0, logx=False)


ax[0].set_ylim(3e-17, 4e-15)
ax[1].set_ylim(7e-16, 4e-15)
ax[1].set_xlabel(r"Rest frame Wavelength [\AA]")
fig.text(
    0.018, 0.5, r"Flux Density [erg s$^{-1}$ cm$^{-2}$ $\AA^{-1}$]", rotation="vertical", ha="center",
    va="center"
)

fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
fig.subplots_adjust(hspace=0)
fig.savefig("../paper_figures/figure10_tde_14-li_2019qiz_models.pdf", dpi=300)
# fig.savefig("../paper_figures/tde_14-li_2019qiz_models.pdf", dpi=300)

plt.show()
