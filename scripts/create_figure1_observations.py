#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from pypython.util import smooth_array
from pypython.plotutil import ax_add_line_ids
from pypython.util import get_array_index
from pypython import plotutil

plotutil.set_default_rcparams()

lw = 2
alp = 0.75
sm = 5
norm = 3e-16

# indices for spectrum arrays, do not touch!!!
x1 = 0
x2 = -1

important_lines = [
    [r"Ly$~\alpha$/N \textsc{v}", 1216],
    ["", 1240],
    [r"O \textsc{v}/Si \textsc{iv}", 1371],
    ["", 1400],
    [r"C \textsc{iv}", 1549],
    [r"He \textsc{ii}", 1640],
    [r"N \textsc{iii]} / H$~\delta$", 4100],
    [r"H$~\gamma$", 4340],
    [r"He \textsc{ii}", 4686],
    [r"H$~\beta$", 4861],
    [r"Na \textsc{i}", 5891],
    [r"H$~\alpha$", 6564],
]

opt_assasn14li = np.loadtxt("../observed_spectra/figure/asassn14li_52d.dat", delimiter=",", skiprows=1)
opt_at2018zr = np.loadtxt("../observed_spectra/figure/AT2018zr_2018-03-30_07-35-47_WHT-4.2m_ACAM_TNS.ascii")
opt_iptf15af = np.loadtxt("../observed_spectra/figure/iPTF15af_2015-03-23_07-16-37_Keck1_LRIS_iPTF.ascii")
uv_assasn14li = np.loadtxt("../observed_spectra/figure/ASASSN-14li_spec_Cenko.dat")
uv_at2018zr = np.loadtxt("../observed_spectra/figure/at2018zr_59d.dat")
uv_iptf15af = np.loadtxt("../observed_spectra/figure/Blagorodnova_iPTF15af.dat")

opt_objects = [
    opt_iptf15af, opt_at2018zr, opt_assasn14li
]
uv_objects = [
    uv_iptf15af, uv_at2018zr, uv_assasn14li
]
obj_name = [
    r"iPTF15af", r"AT2018zr", r"ASASSN-14li"
]
colors = [
    "C0", "C1", "C0"
]
uv_rescale = [
    1, 15, 75
]
opt_rescale = [
    0.5, 1, 200
]
uv_label = [
    r"$\Delta t_{\rm UV} =$ 52d", r"$\Delta t_{\rm UV} =$ 59d", r"$\Delta t_{\rm UV} =$ 60d"
]
label_ypos = [
    0.4, 14, 1000,
]
sm_opt = [
    sm, sm, 20
]
opt_label = [
    r"$\Delta t_{\rm opt} =$ 67d",
    r"$\Delta t_{\rm opt} =$ 12d",  # TODO: try to find a better replacement for this spectrum
    r"$\Delta t_{\rm opt} =$ 52d"
]
redshift = [
    0.07897,   # iPTF15af      TDE-Bowen
    0.071,     # AT2018zr      TDE-H
    0.0206,    # ASASSN-14li   TDE-Bowen
]

t_ax = []
fig = plt.figure(figsize=(11, 14))
gs = fig.add_gridspec(5, 1)
ax1 = fig.add_subplot(gs[:2, :])
ax2 = fig.add_subplot(gs[2:, :])
t_ax = [ax1, ax2]

# ##############################################################################
#
#                               OPTICAL + UV panel
#
# ##############################################################################

ax = t_ax[0]

for n in range(len(uv_objects)):
    # PLOT UV spectra
    x = uv_objects[n][:, 0] / (redshift[n] + 1)
    y = uv_objects[n][:, 1]
    ax.plot(x[x1:x2], smooth_array(y[x1:x2], sm) / norm * uv_rescale[n], alpha=alp, color=colors[n])
    # PLOT OPT spectra
    x = opt_objects[n][:, 0] / (redshift[n] + 1)
    y = opt_objects[n][:, 1]
    x1 = get_array_index(x, 3250)
    if x1 == -1:
        x1 = 0
    x2 = get_array_index(x, 7800)
    ax.plot(x[x1:x2], smooth_array(y[x1:x2], sm_opt[n]) / norm * opt_rescale[n], alpha=alp, color=colors[n])
    # LABEL the plot
    string = "{}\n{} rescaled with {}".format(obj_name[n], uv_label[n], uv_rescale[n]) + r"$F_{\rm norm}$" + \
        "\n{} rescaled with {}".format(opt_label[n], opt_rescale[n]) + r"$F_{\rm norm}$"
    ax.text(3500, label_ypos[n], string, color=colors[n], fontsize=13)

ax.text(
    1000, 0.15, r"$F_{\rm norm}$ =" + r" {:3.0e}".format(norm) + r" [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", color="k",
    fontsize=13
)

ax.set_xscale("log")  # , subsx=[2000, 4000, 6000])
ax.set_yscale("log")
ax.set_ylim(0.1, 2e5)
ax = ax_add_line_ids(ax, important_lines, logx=True, fontsize=13, linestyle="none", offset=0, ynorm=0.87)
ax.set_ylabel(r"Normalized Flux $\times$ constant")

# DO NOT TOUCH THERE IS SOME ARCANE WIZARDY HERE WHICH ALLOWS ME TO HAVE NICE
# X AXIS LABELS ON THIS FIGURE

# fmt = ScalarFormatter()
# fmt.set_scientific(False)
# ax.xaxis.set_major_formatter(fmt)
# ax.xaxis.set_minor_locator(MultipleLocator(1000))
# ax.xaxis.set_minor_formatter(fmt)
# ax.xaxis.set_ticklabels(["1000", "2000", "", "4000", "", "6000", "", "8000"], minor=True)

# ##############################################################################
#
#                              OPTICAL ONLY panel
#
# ##############################################################################

ax = t_ax[1]

lw = 2
alp = 0.75
sm = 5
norm = 0.7e-16

# Load in optical spectra
assasn14li = np.loadtxt("../observed_spectra/figure/asassn14li_52d.dat", delimiter=",", skiprows=1)
assasn14ae = np.loadtxt("../observed_spectra/figure/asassn14ae_60d.dat", delimiter=",", skiprows=1)
at2018zr = np.loadtxt("../observed_spectra/figure/AT2018zr_2018-03-30_07-35-47_WHT-4.2m_ACAM_TNS.ascii")
iptf15af = np.loadtxt("../observed_spectra/figure/iPTF15af_2015-03-23_07-16-37_Keck1_LRIS_iPTF.ascii")
at2019dsg = np.loadtxt("../observed_spectra/figure/AT2019dsg_2019-05-13_08-40-46.020_ESO-NTT_EFOSC2-NTT_ePESSTOp.ascii")
at2018iih = np.loadtxt("../observed_spectra/figure/AT2018iih_2019-03-11_01-01-28_DCT_Deveny-LMI_TNS.ascii")

objects = [
    assasn14li, iptf15af, at2019dsg, assasn14ae, at2018zr, at2018iih
]
name = [
    r"ASASSN-14li $\Delta t = 52$ d", r"iPTF15af $\Delta t = 67$ d", r"AT2019dsg $\Delta t = 13$ d",
    r"ASASSN-14ae $\Delta t = 60$ d", r"AT2018zr $\Delta t = 50$ d", r"AT2018iih $\Delta t = 102$ d",
]
name_y = [
    7.5, 25, 60, 200, 460, 1337
]
colors = [
    "C0", "C0", "C0", "C1", "C1", "C2"
]
renorm = [
    1, 13, 10, 23, 30, 3000,
]
redshift = [
    0.0206,    # ASASSN-14li   TDE-Bowen
    0.07897,   # iPTF15af      TDE-Bowen
    0.0512,    # AT2019dsg     TDE-Bowen
    0.043671,  # ASASSN-14ae   TDE-H
    0.075,     # AT2018zr      TDE-H
    0.212,     # AT2018iih     TDE-He
]
lines = [
    ["O III", 3760],
    [r"N III / H$_{\delta}$", 4100],
    [r"", 4101],
    [r"H$_{\gamma}$", 4340],
    ["N III / He II", 4641],
    ["", 4686],
    [r"H$_{\beta}$", 4861],
    [r"H$_{\alpha}$", 6564],
]

assert(len(redshift) == len(objects)), "The number of redshifts provided does not meet the number of objects plotting"

for n in range(len(objects)):
    x = objects[n][:, 0] / (redshift[n] + 1)
    y = objects[n][:, 1]
    x1 = get_array_index(x, 3500)
    x2 = get_array_index(x, 7000)
    if x1 == -1:
        x1 = 0
    if name[n] == r"ASASSN-14li $\Delta t = 52$ d":
        sm = 20  # Noisy spectrum for some reason
    else:
        sm = 5
    ax.plot(
        x[x1:x2], smooth_array(y[x1:x2], sm) / norm * renorm[n], label=name[n], alpha=alp, color=colors[n],
        zorder=0
    )
    text = ax.text(5000, name_y[n], name[n], color=colors[n], fontsize=15)

ax.text(
    3420, 1.2, r"$F_{\rm norm}$ =" + r" {:3.0e}".format(norm) + r" [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]", color="k",
    fontsize=13
)

ax.set_xscale("linear")
ax.set_yscale("log")
ax.set_ylim(1, 7e3)
ax.set_xlabel(r"Rest frame Wavelength [\AA]")
ax.set_ylabel(r"Normalized Flux $\times$ constant")
ax = ax_add_line_ids(ax, important_lines, logx=False, offset=0, linestyle="none", fontsize=13, ynorm=0.93)

fig.tight_layout(rect=[0.03, 0.03, 0.97, 0.97])
plt.savefig("../paper_figures/figure1_observations.pdf", dpi=300)
# plt.savefig("../paper_figures/optical_uv_tde_observations.png", dpi=300)
plt.close()
