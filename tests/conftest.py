# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "policy(instance): serve this policy from the driver_stub fixture"
    )
