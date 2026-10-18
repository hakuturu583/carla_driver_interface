"""The egodriver contract without a socket: a gRPC channel that calls the servicer.

A runtime that serves its policy from its own process (autoware_carla_scenario's
``driver.policy``) still pays for TCP: every drive's megabytes of LiDAR are copied
through the loopback and HTTP/2 framing twice. :class:`InProcessChannel` is a
:class:`grpc.Channel` that skips the transport and nothing else:

* the stubs are the generated ones (``EgodriverServiceStub``,
  ``CarlaVehicleServiceStub``), and the servicer registers on the channel with the
  generated ``add_*_to_server`` exactly as on a server, so method names and
  (de)serializers are the generated code's;
* every request is serialized to bytes and parsed from them by the servicer's
  deserializer, and every response likewise -- the bytes each side sees are the
  bytes it would see on the wire, never a shared message object;
* the servicer gets a :class:`grpc.ServicerContext` with the semantics it can
  observe on a real server: ``abort`` ends the call with its code and details,
  an exception becomes ``UNKNOWN`` ("Exception calling application: ..."),
  ``set_code`` / ``set_details`` fail a call that returns, invocation metadata,
  ``send_initial_metadata`` (once) and trailing metadata reach the other side
  (the contract revision is negotiated through them,
  :mod:`carla_driver_interface.contract`);
* the client sees a failed call as a :class:`grpc.RpcError` that is also a
  :class:`grpc.Call` (``code()``, ``details()``, both metadata), as grpc raises
  it; an unknown method is ``UNIMPLEMENTED`` ("Method not found!"), an
  oversized request ``RESOURCE_EXHAUSTED``, a call on a closed channel a
  ``ValueError``.

What differs is only what a socket adds: the servicer runs on the calling
thread, so a deadline cannot cut a call short -- a call that overran its
``timeout`` fails with ``DEADLINE_EXCEEDED`` when it returns, its effects done,
as a server that finishes after the client gave up has done them too.

Use :func:`in_process_channel` for a policy, as :func:`~carla_driver_interface.server.build_server`
serves one::

    channel = in_process_channel(policy)
    stub = EgodriverServiceStub(channel)
"""

from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import grpc

from .driver import BaseDriver
from .protocol import (
    MAX_MESSAGE_BYTES,
    add_CarlaVehicleServiceServicer_to_server,
    add_EgodriverServiceServicer_to_server,
    channel_options,
)
from .service import EgodriverServicer, ServiceMode

logger = logging.getLogger(__name__)

__all__ = ["InProcessChannel", "InProcessRpcError", "in_process_channel"]

#: What :meth:`grpc.ServicerContext.peer` answers on this channel.
PEER = "in-process"

Metadata = Tuple[Tuple[str, Any], ...]


class _Metadatum(collections.namedtuple("_Metadatum", ("key", "value"))):
    """One metadata entry, as grpc hands them out (``key``, ``value``, a pair)."""


def _metadata(entries: Optional[Iterable[Tuple[str, Any]]]) -> Metadata:
    """*entries* as grpc delivers them: checked as grpc checks them, then frozen.

    Raises:
        TypeError: A value grpc refuses before sending (not ``str``/``bytes``,
            or ``str`` under a ``-bin`` key).
        InProcessRpcError: ``INTERNAL`` for a key HTTP/2 cannot carry.
    """
    out = []
    for key, value in entries or ():
        if key.endswith("-bin"):
            if not isinstance(value, bytes):
                raise TypeError(
                    f'Binary metadata key="{key}" expected bytes, got {type(value)}'
                )
        elif not isinstance(value, (str, bytes)):
            raise TypeError(f"Expected str, not {type(value)}")
        if not _legal_key(key):
            raise InProcessRpcError(grpc.StatusCode.INTERNAL, "Invalid metadata")
        if isinstance(value, bytes) and not key.endswith("-bin"):
            value = value.decode("ascii")
        out.append(_Metadatum(key, value))
    return tuple(out)


_KEY_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-_.")


def _legal_key(key: str) -> bool:
    # HTTP/2 header names, lower case; grpc reserves the ``grpc-`` prefix.
    return bool(key) and set(key) <= _KEY_CHARS and not key.startswith("grpc-")


class InProcessRpcError(grpc.RpcError, grpc.Call, grpc.Future):
    """A failed call, as grpc raises one: an :class:`grpc.RpcError`, a :class:`grpc.Call`
    and a done :class:`grpc.Future` (an interceptor's continuation returns it as the
    call's outcome)."""

    def __init__(
        self,
        code: grpc.StatusCode,
        details: str,
        initial_metadata: Metadata = (),
        trailing_metadata: Metadata = (),
    ) -> None:
        super().__init__(details)
        self._code = code
        self._details = details
        self._initial_metadata = initial_metadata
        self._trailing_metadata = trailing_metadata

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details

    def initial_metadata(self) -> Metadata:
        return self._initial_metadata

    def trailing_metadata(self) -> Metadata:
        return self._trailing_metadata

    def debug_error_string(self) -> str:
        return f"in-process call failed: {self._code.name}: {self._details}"

    def is_active(self) -> bool:
        return False

    def time_remaining(self) -> Optional[float]:
        return None

    def cancel(self) -> bool:
        return False

    def add_callback(self, callback: Callable[[], None]) -> bool:
        return False

    # -- grpc.Future, already done -------------------------------------------------

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> Any:
        raise self

    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        return self

    def traceback(self, timeout: Optional[float] = None) -> Any:
        return self.__traceback__

    def add_done_callback(self, fn: Callable[[Any], None]) -> None:
        fn(self)

    def __str__(self) -> str:
        return (
            f"<{type(self).__name__} of RPC that terminated with:\n"
            f'\tstatus = {self._code}\n\tdetails = "{self._details}"\n>'
        )

    __repr__ = __str__


class _Aborted(Exception):
    """Raised by :meth:`_ServicerContext.abort`; grpc raises a bare ``Exception``."""


class _ServicerContext(grpc.ServicerContext):
    """The server half of one call: what a servicer can see of grpc's own context."""

    def __init__(self, metadata: Metadata, deadline: Optional[float]) -> None:
        self._invocation_metadata = metadata
        self._deadline = deadline
        self._lock = threading.Lock()
        self.initial_metadata: Optional[Metadata] = None
        self.trailing: Optional[Metadata] = None
        self.status_code: Optional[grpc.StatusCode] = None
        self.status_details: Optional[str] = None
        self.aborted = False
        self._callbacks: List[Callable[[], None]] = []
        self._active = True

    # -- grpc.RpcContext -----------------------------------------------------

    def is_active(self) -> bool:
        return self._active and (self._deadline is None or time.time() < self._deadline)

    def time_remaining(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(self._deadline - time.time(), 0.0)

    def cancel(self) -> None:
        with self._lock:
            self.status_code = grpc.StatusCode.CANCELLED
            self.status_details = "Cancelled"
            self.aborted = True
        raise _Aborted()

    def add_callback(self, callback: Callable[[], None]) -> bool:
        with self._lock:
            if not self._active:
                return False
            self._callbacks.append(callback)
            return True

    # -- grpc.ServicerContext --------------------------------------------------

    def invocation_metadata(self) -> Metadata:
        return self._invocation_metadata

    def peer(self) -> str:
        return PEER

    def peer_identities(self) -> Optional[Iterable[bytes]]:
        return None

    def peer_identity_key(self) -> Optional[str]:
        return None

    def auth_context(self) -> Dict[str, Iterable[bytes]]:
        return {}

    def set_compression(self, compression: grpc.Compression) -> None:
        pass

    def disable_next_message_compression(self) -> None:
        pass

    def send_initial_metadata(
        self, initial_metadata: Iterable[Tuple[str, Any]]
    ) -> None:
        with self._lock:
            if self.initial_metadata is not None:
                raise ValueError("Initial metadata no longer allowed!")
            self.initial_metadata = _metadata(initial_metadata)

    def set_trailing_metadata(
        self, trailing_metadata: Iterable[Tuple[str, Any]]
    ) -> None:
        with self._lock:
            self.trailing = _metadata(trailing_metadata)

    def trailing_metadata(self) -> Optional[Metadata]:
        return self.trailing

    def abort(self, code: grpc.StatusCode, details: str) -> None:
        if code == grpc.StatusCode.OK:
            logger.error("abort() called with StatusCode.OK; returning UNKNOWN")
            code, details = grpc.StatusCode.UNKNOWN, ""
        with self._lock:
            self.status_code = code
            self.status_details = details
            self.aborted = True
        raise _Aborted()

    def abort_with_status(self, status: grpc.Status) -> None:
        self.trailing = _metadata(status.trailing_metadata)
        self.abort(status.code, status.details)

    def set_code(self, code: grpc.StatusCode) -> None:
        with self._lock:
            self.status_code = code

    def code(self) -> Optional[grpc.StatusCode]:
        return self.status_code

    def set_details(self, details: str) -> None:
        with self._lock:
            self.status_details = details

    def details(self) -> Optional[str]:
        return self.status_details

    # -- the call's end ----------------------------------------------------------

    def finish(self) -> None:
        with self._lock:
            self._active = False
            callbacks, self._callbacks = self._callbacks, []
        for callback in callbacks:
            try:
                callback()
            except Exception:  # noqa: BLE001 - grpc logs and carries on too
                logger.exception("a call's termination callback raised")

    def failure(self, code: grpc.StatusCode, details: str) -> InProcessRpcError:
        """The client's error for a call that ended here; the context's own status wins."""
        return InProcessRpcError(
            self.status_code or code,
            details if self.status_details is None else self.status_details,
            self.initial_metadata or (),
            self.trailing or (),
        )


class _Call(grpc.Call):
    """A completed call's outcome as ``with_call`` hands it back."""

    def __init__(self, initial_metadata: Metadata, trailing_metadata: Metadata) -> None:
        self._initial_metadata = initial_metadata
        self._trailing_metadata = trailing_metadata

    def initial_metadata(self) -> Metadata:
        return self._initial_metadata

    def trailing_metadata(self) -> Metadata:
        return self._trailing_metadata

    def code(self) -> grpc.StatusCode:
        return grpc.StatusCode.OK

    def details(self) -> str:
        return ""

    def is_active(self) -> bool:
        return False

    def time_remaining(self) -> Optional[float]:
        return None

    def cancel(self) -> bool:
        return False

    def add_callback(self, callback: Callable[[], None]) -> bool:
        return False


class _Future(grpc.Future, grpc.Call):
    """``future()``'s answer: already done, since the call ran on this thread."""

    def __init__(
        self, response: Any, call: Optional[_Call], error: Optional[InProcessRpcError]
    ):
        self._response = response
        self._call = call
        self._error = error

    def cancel(self) -> bool:
        return False

    def cancelled(self) -> bool:
        return False

    def running(self) -> bool:
        return False

    def done(self) -> bool:
        return True

    def result(self, timeout: Optional[float] = None) -> Any:
        if self._error is not None:
            raise self._error
        return self._response

    def exception(self, timeout: Optional[float] = None) -> Optional[Exception]:
        return self._error

    def traceback(self, timeout: Optional[float] = None) -> Any:
        return None if self._error is None else self._error.__traceback__

    def add_done_callback(self, fn: Callable[[Any], None]) -> None:
        fn(self)

    def _outcome(self) -> grpc.Call:
        return self._error if self._error is not None else self._call  # type: ignore[return-value]

    def initial_metadata(self) -> Metadata:
        return self._outcome().initial_metadata()

    def trailing_metadata(self) -> Metadata:
        return self._outcome().trailing_metadata()

    def code(self) -> grpc.StatusCode:
        return self._outcome().code()

    def details(self) -> str:
        return self._outcome().details()

    def is_active(self) -> bool:
        return False

    def time_remaining(self) -> Optional[float]:
        return None

    def add_callback(self, callback: Callable[[], None]) -> bool:
        return False


class _UnaryUnary(grpc.UnaryUnaryMultiCallable):
    def __init__(
        self,
        channel: "InProcessChannel",
        method: str,
        request_serializer: Optional[Callable[[Any], bytes]],
        response_deserializer: Optional[Callable[[bytes], Any]],
    ) -> None:
        self._channel = channel
        self._method = method
        self._serialize = request_serializer
        self._deserialize = response_deserializer

    def __call__(
        self,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[Sequence[Tuple[str, Any]]] = None,
        credentials: Optional[grpc.CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[grpc.Compression] = None,
    ) -> Any:
        response, _ = self.with_call(request, timeout, metadata)
        return response

    def with_call(
        self,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[Sequence[Tuple[str, Any]]] = None,
        credentials: Optional[grpc.CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[grpc.Compression] = None,
    ) -> Tuple[Any, grpc.Call]:
        return self._channel._invoke(
            self._method, request, self._serialize, self._deserialize, timeout, metadata
        )

    def future(
        self,
        request: Any,
        timeout: Optional[float] = None,
        metadata: Optional[Sequence[Tuple[str, Any]]] = None,
        credentials: Optional[grpc.CallCredentials] = None,
        wait_for_ready: Optional[bool] = None,
        compression: Optional[grpc.Compression] = None,
    ) -> _Future:
        try:
            response, call = self.with_call(request, timeout, metadata)
        except InProcessRpcError as error:
            return _Future(None, None, error)
        return _Future(response, call, None)  # type: ignore[arg-type]


class _HandlerCallDetails(
    collections.namedtuple("_HandlerCallDetails", ("method", "invocation_metadata")),
    grpc.HandlerCallDetails,
):
    pass


def _option(options: Sequence[Tuple[str, Any]], name: str, default: int) -> int:
    value = dict(options).get(name, default)
    return int(value) if int(value) >= 0 else 2**63


class InProcessChannel(grpc.Channel):
    """A :class:`grpc.Channel` whose unary calls run the registered servicers here.

    It is its own server too: a servicer registers on it with the generated
    ``add_*_to_server(servicer, channel)``, as on a :class:`grpc.Server`.

    Args:
        options: Channel options; only the message size limits are read
            (``grpc.max_send_message_length``, ``grpc.max_receive_message_length``),
            :func:`~carla_driver_interface.protocol.channel_options` by default,
            as both ends of this contract use.
    """

    def __init__(self, options: Optional[Sequence[Tuple[str, Any]]] = None) -> None:
        options = channel_options() if options is None else list(options)
        self._max_send = _option(
            options, "grpc.max_send_message_length", MAX_MESSAGE_BYTES
        )
        self._max_receive = _option(
            options, "grpc.max_receive_message_length", MAX_MESSAGE_BYTES
        )
        self._generic_handlers: List[grpc.GenericRpcHandler] = []
        self._registered: Dict[str, grpc.RpcMethodHandler] = {}
        self._closed = False

    # -- the server half ---------------------------------------------------------

    def add_generic_rpc_handlers(
        self, generic_rpc_handlers: Iterable[grpc.GenericRpcHandler]
    ) -> None:
        """Register handlers, as :meth:`grpc.Server.add_generic_rpc_handlers` does."""
        self._generic_handlers.extend(generic_rpc_handlers)

    def add_registered_method_handlers(
        self, service_name: str, method_handlers: Dict[str, grpc.RpcMethodHandler]
    ) -> None:
        """Register handlers by method, as newer generated code also does on a server."""
        for name, handler in method_handlers.items():
            self._registered[f"/{service_name}/{name}"] = handler

    def _handler(
        self, method: str, metadata: Metadata
    ) -> Optional[grpc.RpcMethodHandler]:
        if method in self._registered:
            return self._registered[method]
        details = _HandlerCallDetails(method, metadata)
        for generic in self._generic_handlers:
            handler = generic.service(details)
            if handler is not None:
                return handler
        return None

    # -- one call ----------------------------------------------------------------

    def _invoke(
        self,
        method: str,
        request: Any,
        serialize: Optional[Callable[[Any], bytes]],
        deserialize: Optional[Callable[[bytes], Any]],
        timeout: Optional[float],
        metadata: Optional[Sequence[Tuple[str, Any]]],
    ) -> Tuple[Any, grpc.Call]:
        if self._closed:
            raise ValueError("Cannot invoke RPC on closed channel!")
        invocation = _metadata(metadata)
        started = time.time()
        deadline = None if timeout is None else started + timeout
        if timeout is not None and timeout < 0:
            raise InProcessRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline Exceeded"
            )
        try:
            request_bytes = request if serialize is None else serialize(request)
        except Exception:  # noqa: BLE001 - grpc reports any serializer failure
            raise InProcessRpcError(
                grpc.StatusCode.INTERNAL, "Exception serializing request!"
            ) from None
        if len(request_bytes) > self._max_send:
            raise InProcessRpcError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"CLIENT: Sent message larger than max ({len(request_bytes)} vs. "
                f"{self._max_send})",
            )
        handler = self._handler(method, invocation)
        if handler is None:
            raise InProcessRpcError(grpc.StatusCode.UNIMPLEMENTED, "Method not found!")
        if handler.unary_unary is None:
            raise NotImplementedError(
                f"{method} is not unary-unary; the in-process channel serves unary calls only"
            )

        response_bytes = self._serve(handler, request_bytes, invocation, deadline)
        if deadline is not None and timeout and time.time() > deadline:
            # The servicer ran to the end on this thread; the client gave up.
            raise InProcessRpcError(
                grpc.StatusCode.DEADLINE_EXCEEDED, "Deadline Exceeded"
            )
        context, payload = response_bytes
        initial = context.initial_metadata or ()
        trailing = context.trailing or ()
        if len(payload) > self._max_receive:
            raise InProcessRpcError(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"CLIENT: Received message larger than max ({len(payload)} vs. "
                f"{self._max_receive})",
                initial,
                trailing,
            )
        try:
            response = payload if deserialize is None else deserialize(payload)
        except Exception:  # noqa: BLE001
            raise InProcessRpcError(
                grpc.StatusCode.INTERNAL,
                "Exception deserializing response!",
                initial,
                trailing,
            ) from None
        return response, _Call(initial, trailing)

    def _serve(
        self,
        handler: grpc.RpcMethodHandler,
        request_bytes: bytes,
        invocation: Metadata,
        deadline: Optional[float],
    ) -> Tuple[_ServicerContext, bytes]:
        """The server's half: parse, call, serialize; an error ends it as grpc's would."""
        context = _ServicerContext(invocation, deadline)
        try:
            try:
                request = (
                    request_bytes
                    if handler.request_deserializer is None
                    else handler.request_deserializer(request_bytes)
                )
            except Exception:  # noqa: BLE001
                raise context.failure(
                    grpc.StatusCode.INTERNAL, "Exception deserializing request!"
                ) from None
            try:
                response = handler.unary_unary(request, context)
            except Exception as exception:  # noqa: BLE001 - as grpc's server
                if context.aborted:
                    raise context.failure(
                        grpc.StatusCode.UNKNOWN, "RPC Aborted"
                    ) from None
                details = f"Exception calling application: {exception}"
                logger.exception(details)
                raise context.failure(grpc.StatusCode.UNKNOWN, details) from None
            if context.aborted:
                # A servicer that swallowed its own abort still aborted.
                raise context.failure(grpc.StatusCode.UNKNOWN, "RPC Aborted")
            try:
                payload = (
                    response
                    if handler.response_serializer is None
                    else handler.response_serializer(response)
                )
            except Exception:  # noqa: BLE001
                payload = None
            if payload is None:
                raise context.failure(
                    grpc.StatusCode.INTERNAL, "Failed to serialize response!"
                )
            if context.status_code not in (None, grpc.StatusCode.OK):
                raise context.failure(context.status_code, "")
            return context, payload
        finally:
            context.finish()

    # -- grpc.Channel ------------------------------------------------------------

    def unary_unary(
        self,
        method: str,
        request_serializer: Optional[Callable[[Any], bytes]] = None,
        response_deserializer: Optional[Callable[[bytes], Any]] = None,
        _registered_method: Optional[bool] = False,
    ) -> grpc.UnaryUnaryMultiCallable:
        return _UnaryUnary(self, method, request_serializer, response_deserializer)

    def unary_stream(self, method: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the in-process channel serves unary calls only")

    def stream_unary(self, method: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the in-process channel serves unary calls only")

    def stream_stream(self, method: str, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("the in-process channel serves unary calls only")

    def subscribe(
        self,
        callback: Callable[[grpc.ChannelConnectivity], None],
        try_to_connect: bool = False,
    ) -> None:
        callback(grpc.ChannelConnectivity.READY)

    def unsubscribe(self, callback: Callable[[grpc.ChannelConnectivity], None]) -> None:
        pass

    def close(self) -> None:
        self._closed = True

    def __enter__(self) -> "InProcessChannel":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def in_process_channel(
    driver: BaseDriver, mode: ServiceMode = "carla"
) -> InProcessChannel:
    """A channel serving ``driver`` in this process, as :func:`~carla_driver_interface.server.build_server` serves it.

    The same :class:`~carla_driver_interface.service.EgodriverServicer`, registered
    for both services a server carries (``egodriver.EgodriverService`` and
    ``carla_driver.v0.CarlaVehicleService``).
    """
    channel = InProcessChannel(channel_options())
    servicer = EgodriverServicer(driver, mode)
    add_EgodriverServiceServicer_to_server(servicer, channel)
    add_CarlaVehicleServiceServicer_to_server(servicer, channel)
    return channel
