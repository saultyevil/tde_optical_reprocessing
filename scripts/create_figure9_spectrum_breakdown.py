#!/usr/bin/env python

import pandas as pd
from matplotlib import pyplot as plt
from typing import Union
from pypython.util import smooth_array
from pypython import createspectrum
from pypython import spectrum
from pypython import plotutil
from pypython.physics.convert import angstrom_to_hz
from pypython.physics import constants

DEFAULT_DISTANCE = 100 * constants.PARSEC
SCALED_DISTANCE = 100 * 1e6 * constants.PARSEC
SCALE_FACTOR = DEFAULT_DISTANCE ** 2 / SCALED_DISTANCE ** 2

plotutil.set_default_rcparams()


def plot(
    contributions: dict, output_name: str, xmin: float = None, xmax: float = None, line_labels: bool = False,
    sm: int = 1, alpha: float = 0.7, show: bool = False
) -> Union[plt.Figure, plt.Axes]:
    """Plot the different process spectra"""

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharey="row")

    the_names = ["Full", r"H$\alpha$", r"H$\alpha$ w/ ES", "Continuum"]

    for n, inclination in enumerate(["10", "60"]):

        for (name, s), myname in zip(contributions.items(), the_names):
            ax[n].plot(
                s["Lambda"], smooth_array(s[inclination] * SCALE_FACTOR, sm), label=myname, alpha=alpha
            )

        ax[n].set_yscale("log")
        ax[n].set_xscale("linear")
        ax[n].set_xlim(xmin, xmax)
        ax[n].set_ylim(2e-6 * SCALE_FACTOR, 2e-3 * SCALE_FACTOR)
        ax[n].text(0.93, 0.93, inclination + r"$^{\circ}$", transform=ax[n].transAxes, fontsize=15)
        if line_labels:
            ax[n] = plotutil.ax_add_line_ids(ax[n], plotutil.common_lines(), fontsize=15, offset=3)

    ax[0].legend(loc="lower center", ncol=2)
    ax[0].set_ylabel(r"Flux Density at 100 Mpc [erg s$^{-1}$ cm$^{-2}$ \AA$^{-1}$]")
    fig.text(0.5, 0.03, r"Rest frame Wavelength [\AA]", ha="center", va="center")
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.98])
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(output_name, dpi=300)

    plt.close()

    return fig, ax


def get_spectrum_breakdown(
    root: str, wl_min: float, wl_max: float, n_cores_norm: int = 1, spec_cycle_norm: float = 1, wd: str = ".",
    nres: int = None, mode_line_res: bool = True
) -> dict:
    """Get the spectrum for each physical process"""

    df = createspectrum.read_delay_dump(root, cd=wd, db=False)
    s = spectrum.Spectrum(root, wd)

    # create dataframes for each physical process, what you can actually get
    # depends on mode_line_res, i.e. if LineRes. is included or not. Store these
    # data frame in a list

    contributions = []
    created_spectra = [s]
    contribution_names = ["Extracted"]

    contributions.append(df[df["Res."] == nres])
    contribution_names.append("Res" + str(nres))

    contributions.append(df[df["LineRes."] == nres])
    contribution_names.append("LineRes" + str(nres))

    contribution_names.append("Continuum")
    contributions.append(
        pd.concat(
            [
                df[df["LineRes."] == -1],
                df[df["Res."] == -2],
                df[df["Res."] > 20000]
            ]
        )
    )

    n_spec = len(contributions) + 1

    for contribution in contributions:
        created_spectra.append(
            createspectrum.create_spectrum(
                root, wd, dumped_photons=contribution, freq_min=angstrom_to_hz(wl_max), freq_max=angstrom_to_hz(wl_min),
                n_cores_norm=n_cores_norm, spec_cycle_norm=spec_cycle_norm
            )
        )

    return {contribution_names[i]: created_spectra[i] for i in range(n_spec)}


if __name__ == "__main__":

    contributions = get_spectrum_breakdown(
        "tde_opt_cmf_lines", 4000, 7000, 48, 1, "../tests/spectrum_contribution/H_alpha_Fid_paper", 430
    )

    fig, ax = plot(
        contributions, "../paper_figures/figure9_spectrum_breakdown.pdf", 6220, 6880, show=True
    )
