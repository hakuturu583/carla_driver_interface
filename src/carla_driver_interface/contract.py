"""The contract revision: what the egodriver wire means beyond alpasim's proto.

alpasim's proto fixes the messages, not every convention inside them, and
carla-driver-interface 1.x read one of them differently from alpasim. Each side
therefore declares the revision it speaks in gRPC metadata (:data:`METADATA_KEY`),
so a client and a server of different releases -- in different processes, where
no package range can see both -- still agree on what they exchange:

====  ==========================  ==============================================
rev   carla-driver-interface      ``AvailableCamera.rig_to_camera``
====  ==========================  ==============================================
1     1.x                         the camera **body** in the rig (x along the
                                  optical axis, y left, z up)
2     2.x                         the camera's **optical** frame in the rig (x
                                  right, y down, z along the optical axis), as
                                  alpasim's renderers declare it
====  ==========================  ==============================================

* A server answers ``get_version`` with its revision in the initial metadata.
  One that does not (carla-driver-interface 1.x) speaks revision 1.
* A client sends its revision with ``start_session``. One that does not speaks
  revision 1 in carla mode (autoware_carla_scenario 4.x) and alpasim's own
  convention, revision 2, in alpasim mode (an upstream alpasim runtime).
* The servicer hands a policy revision :data:`CONTRACT_REVISION` whatever the
  client speaks (:func:`cameras_to_revision`), and a client declares its cameras in
  the revision the server speaks (:func:`negotiate`), so every release pairs with
  every other.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple

import grpc

from .geometry import Pose, body_to_optical, optical_to_body
from .protocol import AvailableCamera, EgodriverServiceStub, Empty, VersionId

logger = logging.getLogger(__name__)

__all__ = [
    "CONTRACT_REVISION",
    "LEGACY_REVISION",
    "METADATA_KEY",
    "ContractError",
    "camera_to_revision",
    "cameras_to_revision",
    "client_revision",
    "contract_metadata",
    "negotiate",
    "read_revision",
]

#: The revision this release speaks.
CONTRACT_REVISION = 2
#: The revision of a peer that declares none: carla-driver-interface 1.x.
LEGACY_REVISION = 1
#: Every revision this release can translate from and to.
SUPPORTED_REVISIONS: Tuple[int, ...] = (1, 2)
#: gRPC metadata key carrying the revision (keys are lower case).
METADATA_KEY = "carla-driver-interface-contract"


class ContractError(ValueError):
    """A peer declared a revision this release cannot translate."""


def contract_metadata(revision: int = CONTRACT_REVISION) -> Tuple[Tuple[str, str], ...]:
    """gRPC metadata declaring *revision*."""
    return ((METADATA_KEY, str(revision)),)


def read_revision(
    metadata: Optional[Iterable[Tuple[str, object]]], default: int
) -> int:
    """The revision declared in *metadata*, or *default* when it declares none.

    Raises:
        ContractError: The declared revision is not one this release supports.
    """
    for key, value in metadata or ():
        if key != METADATA_KEY:
            continue
        text = value.decode() if isinstance(value, bytes) else str(value)
        try:
            revision = int(text)
        except ValueError:
            raise ContractError(f"{METADATA_KEY}={text!r} is not a revision") from None
        _check_supported(revision)
        return revision
    return default


def camera_to_revision(
    camera: AvailableCamera, from_revision: int, to_revision: int
) -> AvailableCamera:
    """*camera* as revision *to_revision* declares it; the input is left as is."""
    _check_supported(from_revision)
    _check_supported(to_revision)
    converted = AvailableCamera()
    converted.CopyFrom(camera)
    if from_revision == to_revision:
        return converted
    pose = Pose.from_proto(camera.rig_to_camera)
    # Only rig_to_camera differs between revisions 1 and 2.
    pose = optical_to_body(pose) if to_revision == 1 else body_to_optical(pose)
    converted.rig_to_camera.CopyFrom(pose.to_proto())
    return converted


def cameras_to_revision(
    cameras: Iterable[AvailableCamera], from_revision: int, to_revision: int
) -> List[AvailableCamera]:
    """Every camera through :func:`camera_to_revision`."""
    return [camera_to_revision(c, from_revision, to_revision) for c in cameras]


def negotiate(
    stub: EgodriverServiceStub, timeout: Optional[float] = None
) -> Tuple[VersionId, int]:
    """Call ``get_version`` and return the answer with the revision the server speaks.

    A client then declares its cameras with
    ``cameras_to_revision(cameras, CONTRACT_REVISION, revision)`` and sends
    ``contract_metadata(revision)`` with ``start_session``.

    Raises:
        ContractError: The server speaks a revision this release cannot translate;
            the message says what to upgrade.
    """
    version, call = stub.get_version.with_call(Empty(), timeout=timeout)
    try:
        revision = read_revision(call.initial_metadata(), LEGACY_REVISION)
    except ContractError as error:
        raise ContractError(
            f"policy {version.version_id!r}: {error}; upgrade carla-driver-interface "
            f"on this side (it speaks revisions {SUPPORTED_REVISIONS})"
        ) from None
    if revision != CONTRACT_REVISION:
        logger.info(
            "policy %r speaks contract revision %d; translating from %d",
            version.version_id,
            revision,
            CONTRACT_REVISION,
        )
    return version, revision


def client_revision(context: grpc.ServicerContext, default: int) -> int:
    """The revision the client of *context* declared, or *default*."""
    return read_revision(context.invocation_metadata(), default)


def _check_supported(revision: int) -> None:
    if revision not in SUPPORTED_REVISIONS:
        raise ContractError(
            f"contract revision {revision} is not one of {SUPPORTED_REVISIONS}"
        )
