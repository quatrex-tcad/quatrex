# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the SCBA device class for electronic transport calculations."""

from quatrex.core.config import QuatrexConfig
from quatrex.device.base import BaseDevice

# from quatrex.contact.scba import SCBAContact


class SCBADevice(BaseDevice):
    """A quantum device for electronic transport calculations.

    Parameters
    ----------
    config : QuatrexConfig
        Configuration object containing input paths, device parameters,
        and computational settings.

    Attributes
    ----------
    config : QuatrexConfig
        Reference to the configuration object.
    orbital_coordinates : NDArray
        Array of orbital coordinates.
    atom_coordinates : NDArray
        Array of atomic coordinates.
    atomic_species : NDArray
        Array of atom symbols for each atom. NOTE: This array is always on the
        host since CuPy does not support string arrays.
    orbital_offsets : NDArray
        Array of cumulative orbital counts, used to map from atoms to
        orbitals. orbital_offsets[i] gives the starting orbital index
        for atom i.
    potential : NDArray, optional
        Array of electrostatic potential for each orbital.
        Can be either None if no potential is provided or a 1D array
        where the 1D index corresponds to the orbital index or the
        atom index depending on the shape of the provided potential.
    contacts : list[SCBAContact]
        List of Contact objects representing the semi-infinite leads
        connected to this device.

    """

    def __init__(self, config: QuatrexConfig) -> None:
        super().__init__(config)
