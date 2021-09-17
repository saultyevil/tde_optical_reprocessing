#!/usr/bin/env python3

from typing import List, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from pypython import Wind
from pypython.constants import *
from pypython.spectrum.create import wind_bin_interaction_weight


def sightline_coords(x: np.ndarray, theta: float):
    """Return the vertical coordinates for a sightline given the x coordinates
    and the inclination of the sightline.
    """
    return x * np.tan(np.pi / 2 - theta)


def wind_plot(root: str,
              cd: str,
              wind_names: List[str],
              wind_types: List[str],
              subplot_dims: Tuple[int, int],
              fig_size: Tuple[int, int],
              plot_title: str = None) -> None:
    """Actual main plotting routine of the script."""

    if subplot_dims[0] * subplot_dims[1] < len(wind_names):
        print("not enough subplot panels to plot all the provided wind variables")
        return

    fig, ax = plt.subplots(subplot_dims[0],
                           subplot_dims[1],
                           figsize=fig_size,
                           squeeze=False,
                           sharex="col",
                           sharey="row")

    inclinations = ["10", "35", "60", "75", "85"]
    lstyle = ["k-", "k--", "k-.", "k:", "ko-"]
    subplot_titles = [
        r"$\log_{10}$(Electron temperature) [K]", r"$\log_{10}$(Hydrogen density) [cm$^{-3}$]",
        r"$\log_{10}$(Ionization parameter) [cm$^{-3}$]", r"$\log_{10}$(H I fraction) [cm$^{-3}$]",
        r"$\log_{10}$(Polodial velocity) [km s$^{-1}$]", r"$\log_{10}$(Rotational velocity) [km s$^{-1}$]"
    ]

    w = Wind(root, cd, version="84g")

    index = 0
    for i in range(subplot_dims[0]):
        for j in range(subplot_dims[1]):
            if index > len(wind_names) - 1:
                break

            wind_name = wind_names[index]
            wind_type = wind_types[index]

            if wind_name == "H_i01":
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(w["x"],
                                             w["z"],
                                             np.log10(w.get("H_i01f")),
                                             zorder=0,
                                             shading="auto",
                                             vmin=-10)
                try:
                    count_hist = np.loadtxt("../etc/dump/no_partial/tde_opt_dump_wind_Res430_count.txt")
                except IOError:
                    weight_hist, count_hist = wind_bin_interaction_weight("tde_opt_dump", 430, "../etc/dump/no_partial", 4)
                count_hist = np.ma.masked_where(w["inwind"] != 0, count_hist)
                count_hist /= np.sum(count_hist)
                ax[i, j].contour(w["x"], w["z"], count_hist, 3, cmap="plasma")
            elif wind_name == "ne":
                with np.errstate(divide="ignore"):
                    ne = w["H"]["density"]["i01"] + w["H"]["density"]["i02"]
                    im = ax[i, j].pcolormesh(
                        w["x"],
                        w["z"],
                        np.log10(ne),
                        zorder=0,
                        shading="auto",
                    )
            elif wind_name == "v_l" or wind_name == "v_rot":
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(w["x"],
                                             w["z"],
                                             np.log10(w[wind_name]),
                                             zorder=0,
                                             shading="auto",
                                             vmin=1,
                                             vmax=5)
            else:
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(w["x"], w["z"], np.log10(w[wind_name]), zorder=0, shading="auto")

            # print(w.get(wind_name))

            print(wind_name, wind_type)

            fig.colorbar(im, ax=ax[i, j])  # , orientation="horizontal")

            ax[i, j].set_xlim(3e12, np.max(w["x"]))
            ax[i, j].set_ylim(3e12, np.max(w["z"]))
            ax[i, j].set_xscale("log")
            ax[i, j].set_yscale("log")

            if i == 0 and j == 0:
                for k in range(len(inclinations)):
                    if lstyle[k] == "ko-":
                        xsight = np.logspace(np.log10(10), np.log10(np.max(w["x"])), int(30))
                    else:
                        xsight = np.linspace(0, np.max(w["x"]), int(1e5))
                    zsight = sightline_coords(xsight, np.deg2rad(float(inclinations[k])))
                    ax[i, j].plot(xsight, zsight, lstyle[k], label=inclinations[k] + r"$^{\circ}$")
                ax[i, j].legend(loc="lower right")

            ax[i, j].text(0.03,
                          0.93,
                          subplot_titles[index],
                          ha="left",
                          va="center",
                          rotation="horizontal",
                          transform=ax[i, j].transAxes,
                          fontsize=15)

            index += 1

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.text(0.5, 0.02, r"$r$ [cm]", ha="center", va="center", rotation="horizontal")
    fig.text(0.025, 0.5, r"$z$ [cm]", ha="center", va="center", rotation="vertical")

    if plot_title:
        fig.suptitle(plot_title)
    fig.savefig("../p_figures/figure4_wind_properties.pdf", dpi=300)
    plt.show()

    return


root = "tde_opt_spec"
output_name = "wind"
projection = "rectilinear"
path = "../3e6/Mdot_acc/0_15"

wind_names = ["t_e", "ne", "ip", "H_i01", "v_l", "v_rot"]
wind_types = ["wind", "wind", "wind", "ion_density", "wind", "wind"]

wind_plot(root, path, wind_names, wind_types, subplot_dims=(3, 2), fig_size=(13, 15))
