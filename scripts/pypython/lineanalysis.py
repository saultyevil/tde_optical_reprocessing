#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Functions to analyse spectral lines.
"""

import numpy as np
from matplotlib import pyplot as plt
from typing import Union, Tuple
from .extrautil.error import EXIT_FAIL
from .util import get_array_index
from .plotutil import get_y_lims_for_x_lims, ax_add_line_ids, common_lines


def fit_gaussian():
    """Fit a Gaussian to a line profile."""
    return NotImplementedError


def measure_equivalent_width(
    wavelength: np.ndarray, flux: np.ndarray, display_xmin: float, display_xmax: float, ret_fit: bool = False
) -> Union[float, Tuple[float, np.poly1d]]:
    """Measure the equivalent width for an emission or absorption line. A matplotlib
    window will pop up, allowing the user to click on either side of the line
    feature where it ends. A continuum is then fit using a linear fit."""

    coords = []

    def onclick(event):
        """Click twice on a matplotlib figure and record the x coordinates."""
        x = event.xdata
        y = event.ydata
        coords.append(x)
        ax.plot(x, y, "x")
        fig.canvas.draw()
        fig.canvas.flush_events()
        if len(coords) == 2:
            fig.canvas.mpl_disconnect(cid)
            plt.close()
        return coords

    # If the wavelength array looks like it's been passed in descending order,
    # then reverse it to make it in ascending order. This also assumes the
    # flux and continuum flux were also in the same order, so reverses them
    # as well

    if wavelength[1] < wavelength[0]:
        wavelength = wavelength[::-1]
        flux = flux[::-1]

    # There are sanity checks to make sure all array values are in ascending
    # order before we go on any further

    is_increasing = np.all(np.diff(wavelength) > 0)
    if not is_increasing:
        raise ValueError("the values for the wavelength bins provided are not increasing")

    # Plot the spectrum, then allow the user to click on the edges of the line
    # to label where it starts and stops

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.loglog(wavelength, flux, linewidth=2, label="Spectrum")
    ax.set_xlim(display_xmin, display_xmax)
    ax.set_ylim(get_y_lims_for_x_lims(wavelength, flux, display_xmin, display_xmax, scale=2))
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    ax = ax_add_line_ids(ax, common_lines(), logx=True)
    ax.set_title("Mark the blue and then red end of the line")
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()

    # Extract a portion of the spectrum to both the left and right of the line,
    # by 500 Angstroms, this is used to create a linear fit to estimate the
    # continuum

    if len(coords) == 0:
        print("you didn't click on anything!")
        exit(EXIT_FAIL)

    if coords[0] > coords[1]:
        tmp = coords[0]
        coords[0] = coords[1]
        coords[1] = tmp

    wavelength = np.array(wavelength)
    flux = np.array(flux)

    i1 = get_array_index(wavelength, coords[0] - 500)
    j1 = get_array_index(wavelength, coords[0])
    i2 = get_array_index(wavelength, coords[1])
    j2 = get_array_index(wavelength, coords[1] + 500)

    a = np.concatenate([wavelength[i1:j1], wavelength[i2:j2]])
    b = np.concatenate([flux[i1:j1], flux[i2:j2]])
    fit = np.poly1d(np.polyfit(a, b, 1))

    # Plot the spectrum and the fit to see how well it's doing, although if it's
    # a shit fit I do nothing about it and just let the code run its course :-)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.loglog(wavelength, flux, linewidth=2, label="Spectrum")
    # ax.plot(a, b, linewidth=2, label="Extracted bit")
    ax.plot(a, fit(a), label="Linear fit")
    ax.set_xlim(display_xmin, display_xmax)
    ax.set_ylim(get_y_lims_for_x_lims(wavelength, flux, display_xmin, display_xmax, scale=2))
    ax.legend()
    ax.set_xlabel("Wavelength")
    ax.set_ylabel("Flux")
    plt.show()

    # Restrict the wavelength range to be around the line we are interested in

    i_line = get_array_index(wavelength, coords[0])
    j_line = get_array_index(wavelength, coords[1])

    wavelength_ew = wavelength[i_line:j_line]
    flux_ew = flux[i_line:j_line]
    flux_continuum_ew = fit(wavelength_ew)

    # Now we can calculate the equivalent width, remember the formula for this is,
    # W = \int_{\lambda_1}^{\lambda_2} \frac{F_{c} - F_{\lambda}}{F_{c}} d\lambda
    # W = Sum(Fc - Flamda / Fc * dlambda)

    # todo: see if this can be speeded up, but n_bins is usually "small" so
    #       probably this doesn't matter

    w = 0
    n_bins = len(wavelength_ew)
    for i in range(1, n_bins):
        d_wavelength = wavelength_ew[i] - wavelength_ew[i - 1]
        w += (flux_continuum_ew[i] - flux_ew[i]) / flux_continuum_ew[i] * d_wavelength

    # Take the absolute value, as can be negative depending on if the line is in
    # emission or absorption

    w = np.abs(w)

    if ret_fit:
        return w, fit
    else:
        return w


def full_width_half_maximum():
    """Calculate the full width had maximum (FWHM) of a spectral line"""
    def midpoints():
        raise NotImplementedError
    raise NotImplementedError


def line_centroid():
    raise NotImplementedError
