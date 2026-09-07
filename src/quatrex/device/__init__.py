# Copyright (c) 2024-2026 ETH Zurich and the authors of the quatrex package.

"""Includes the device and contact classes."""

from quatrex.core.config import QuatrexConfig
from quatrex.device.base import BaseDevice
from quatrex.device.qtbm import QTBMDevice
from quatrex.device.scba import SCBADevice


def create_device(
    config: QuatrexConfig,
    validate_contacts: bool = True,
) -> QTBMDevice | SCBADevice:
    if config.formalism == "wf":
        device = QTBMDevice(config)
    elif config.formalism == "negf":
        device = SCBADevice(config)
    else:
        raise ValueError(f"Unknown formalism {config.formalism}")

    if validate_contacts:
        device._validate_contacts()

    return device


__all__ = ["create_device", "BaseDevice", "QTBMDevice", "SCBADevice"]
