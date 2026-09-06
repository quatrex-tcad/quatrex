# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the SCBA contact class."""

from qttools import NDArray, xp
from quatrex.contact.base import BaseContact


def order_vector(
    vector: NDArray,
    order: str | NDArray | None,
):
    """Reorders the elements of the given vector according to the
    specified order.

    Parameters
    ----------
    vector : NDArray
        The vector to reorder.
    order : str | NDArray | None
        The order in which to reorder the elements. The only supported
        string is "reverse", which reverses the order of the elements.

    Returns
    -------
    NDArray
        The reordered vector.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return vector
    if order == "reverse":
        return xp.flip(vector, axis=-1)
    return vector[..., order]


def order_block(
    block: NDArray,
    order: str | NDArray | None,
) -> NDArray:
    """Reorders the blocks of the given matrix according to the
    specified order.

    Parameters
    ----------
    block : NDArray
        The matrix block to reorder.
    order : str | NDArray | None
        The order in which to reorder the blocks. The only supported
        string is "reverse", which reverses the order of the blocks.

    Returns
    -------
    NDArray
        The reordered matrix block.

    """

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return block
    if order == "reverse":
        return xp.flip(block, axis=(-2, -1))
    return block[..., :, order][..., order, :]


def get_inverse_order(
    order: str | NDArray | None,
) -> str | NDArray | None:
    """Computes the inverse of the given order.

    Parameters
    ----------
    order : str | NDArray | None

    Returns
    -------
    str | NDArray | None
        The inverse order, or None if the input order is None.

    """
    # TODO: This should be only called once inside
    # the contact.

    if isinstance(order, str) and order not in ["reverse"]:
        raise ValueError(f"Invalid order string: {order}. Must be 'reverse' or None.")
    if isinstance(order, xp.ndarray) and order.ndim != 1:
        raise ValueError(f"Order array must be 1-dimensional, got shape {order.shape}.")

    if order is None:
        return None
    if order == "reverse":
        return "reverse"
    return xp.argsort(order)


class SCBAContact(BaseContact):
    """Class representing a contact for QTBM calculations.

    Parameters
    ----------
    device : BaseDevice
        The device object to which this contact is attached. Contains
        the Hamiltonian, overlap matrices, and atomic structure
        information.
    contact_config : ContactConfig
        The configuration object containing the contact settings such as
        lattice vectors, origin, transport direction, and Fermi level
        information.

    Attributes
    ----------
    device : BaseDevice
        The device object to which this contact is attached.
    name : str
        The contact identifier.
    transport_direction : int
        Transport direction index (0, 1, or 2).
    unit_cell_orbital_indices : dict
        Dict of orbital indices for each contact cell indexed by (i, j,
        k) tuples.
    origin_key : tuple[int, int, int]
        The key corresponding to the origin cell in the
        unit_cell_orbital_indices.
    transverse_repetition_grid: NDArray
        Number of periodic repetitions in the two transverse directions.
    transport_repetitions : int
        Number of repetitions needed in transport direction for
        convergence.
    orbital_indices : NDArray
        Flattened array of orbital indices for the contact, sorted first
        in transport direction, then in transverse directions.
    orbital_indices_per_layer : list[NDArray]
        List of orbital indices for each layer in the transport
        direction, sorted first in transverse directions, then in
        transport direction.
    transverse_to_transport_indices : NDArray
        Indices to reorder the coupling matrix from transverse-first to
        transport-first ordering.
    fermi_level : float
        Fermi level of the contact in eV.
    mid_gap_energy : float
        Mid-gap energy of the contact in eV.
    conduction_band_edge : float
        Energy of the conduction band edge in eV.
    voltage : float
        Voltage applied to the contact in V.
    temperature : float
        Temperature of the contact in K.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
