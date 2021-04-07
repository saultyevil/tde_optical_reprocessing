#!/usr/bin/env python3

from typing import List, Tuple, Union
import numpy as np
from matplotlib import pyplot as plt
from pypython.physics.constants import *
from pypython import wind
from pypython import plotutil
from pypython.createspectrum import wind_bin_interaction_weight

plotutil.set_default_rcparams()


def renormalize_vector(
    a: np.ndarray, scalar: Union[float, int]
) -> np.ndarray:
    """
    This function is used to renormalise a 3-vector quantity.

    Parameters
    ----------
    a:  np.ndarray
        The 3-vector to renormalise.
    scalar: Union[float, int]
        The desired length of the renormalised 3-vector.

    Returns
    -------
    a: np.ndarray
        The renormalised 3-vector quantity.
    """

    eps = 1e-10

    x = np.dot(a, a)

    if x < eps:
        print("Cannot renormalize a vector with magnitude < " + str(eps))
        return -1

    x = scalar / np.sqrt(x)
    a[0] *= x
    a[1] *= x
    a[2] *= x

    return a


def project_cartesian_to_cylindrical_coordinates(
        a: Union[np.ndarray, List[float]], b: Union[np.ndarray, List[float]]
) -> np.ndarray:
    """
    Attempt to a vector from cartesian into cylindrical coordinates.

    Parameters
    ----------
    a: np.ndarray
        The position of the vector in cartesian coordinates.
    b: np.ndarray
        The vector to project into cylindrical coordinates (also in cartesian
        coordinates).

    Returns
    -------
    result: np.ndarray
        The input vector b which is now projected into cylindrical coordinates.
    """

    result = np.zeros(3)
    n_rho = np.zeros(3)
    n_z = np.zeros(3)

    n_rho[0] = a[0]
    n_rho[1] = a[1]
    n_rho[2] = 0

    rc = renormalize_vector(n_rho, 1.0)
    if type(rc) == int:
        return rc

    n_z[0] = n_z[1] = 0
    n_z[2] = 1

    n_phi = np.cross(n_z, n_rho)

    result[0] = np.dot(b, n_rho)
    result[1] = np.dot(b, n_phi)
    result[2] = b[2]

    return result


def sightline_coords(x: np.ndarray, theta: float):
    """
    Return the vertical coordinates for a sightline given the x coordinates
    and the inclination of the sightline.

    Parameters
    ----------
    x: np.ndarray[float]
        The x-coordinates of the sightline
    theta: float
        The opening angle of the sightline

    Returns
    -------
    z: np.ndarray[float]
        The z-coordinates of the sightline
    """

    return x * np.tan(np.pi / 2 - theta)


def wind_plot(
    root: str, cd: str, wind_names: List[str], wind_types: List[str], wd: str, subplot_dims: Tuple[int, int],
    fig_size: Tuple[int, int], plot_title: str = None
) -> None:

    if subplot_dims[0] * subplot_dims[1] < len(wind_names):
        print("not enough subplot panels to plot all the provided wind variables")
        return

    fig, ax = plt.subplots(
        subplot_dims[0], subplot_dims[1], figsize=fig_size, squeeze=False, sharex="col", sharey="row"
    )

    inclinations = ["10", "35", "60", "75", "85"]
    lstyle = ["k-", "k--", "k-.", "k:", "ko-"]
    subplot_titles = [
        r"$\log_{10}$(Electron Temperature) [K]",
        r"$\log_{10}$(Electron Density) [cm$^{-3}$]",
        r"$\log_{10}$(Ionization Parameter) [cm$^{-3}$]",
        r"$\log_{10}$(H I Density) [cm$^{-3}$]",
        r"$\log_{10}$(Polodial Velocity) [km s$^{-1}$]",
        r"$\log_{10}$(Rotational Velocity) [km s$^{-1}$]"
    ]

    w = wind.Wind2D(root, cd)

    index = 0
    for i in range(subplot_dims[0]):
        for j in range(subplot_dims[1]):
            if index > len(wind_names) - 1:
                break

            wind_name = wind_names[index]
            wind_type = wind_types[index]

            if wind_name == "H_i01":
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(
                        w["x"], w["z"], np.log10(w["H"]["density"]["i01"]), zorder=0, shading="auto", vmin=-12,
                        vmax=12
                    )
                weight_hist, count_hist = wind_bin_interaction_weight(root, 430, cd, 48)
                count_hist /= np.sum(count_hist)
                ax[i, j].contour(w["x"], w["z"], count_hist, 3, cmap="plasma")
            elif wind_name == "ne":
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(
                        w["x"], w["z"], np.log10(w[wind_name]), zorder=0, shading="auto", vmin=-12, vmax=12
                    )
            elif wind_name == "v_l" or wind_name == "v_rot":
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(
                        w["x"], w["z"], np.log10(w[wind_name]), zorder=0, shading="auto", vmin=1, vmax=5
                    )
            else:
                with np.errstate(divide="ignore"):
                    im = ax[i, j].pcolormesh(w["x"], w["z"], np.log10(w[wind_name]), zorder=0, shading="auto")

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

            ax[i, j].text(
                0.03, 0.93, subplot_titles[index], ha="left", va="center", rotation="horizontal",
                transform=ax[i, j].transAxes, fontsize=15
            )

            index += 1

    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.text(0.5, 0.02, r"$r$ [cm]", ha="center", va="center", rotation="horizontal")
    fig.text(0.025, 0.5, r"$z$ [cm]", ha="center", va="center", rotation="vertical")

    if plot_title:
        fig.suptitle(plot_title)
    fig.savefig("../paper_figures/figure4_wind_properties.pdf", dpi=300)
    # fig.savefig("../paper_figures/fiducial_wind.png", dpi=300)
    plt.close()

    return


root = "tde_opt_cmf_lines"
output_name = "wind"
projection = "rectilinear"
path = "../model_grids/new_grid_cmf_spec_lines/3e6/Vinf/0_3"

wind_names = ["t_e", "ne", "ip", "H_i01", "v_l", "v_rot"]
wind_types = ["wind", "wind", "wind", "ion_density", "wind", "wind"]

wind_plot(
    root, path, wind_names, wind_types, path, subplot_dims=(3, 2), fig_size=(13, 15)
)
