# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from . import _logger

import torch
import warnings

from enum import IntFlag
from typing import Optional


class _FallbackPolicy(IntFlag):
    """Policy to trigger fallback from ONNX Runtime engine to PyTorch

    Each policy can be combined with the others (using |) in order to aggregate them"""

    FALLBACK_DISABLE = 1
    FALLBACK_FORCE_TORCH_FORWARD = 2
    FALLBACK_UNSUPPORTED_DEVICE = 4
    FALLBACK_UNSUPPORTED_INPUT = 8
    FALLBACK_UNSUPPORTED_OUTPUT = 16
    FALLBACK_UNSUPPORTED_TORCH_MODEL = 32
    FALLBACK_UNSUPPORTED_ONNX_MODEL = 64

    def is_set(self, policy):
        '''Check whether `policy` is set on the `_FallbackPolicy instance

        FALLBACK_DISABLE implies the check will always fail and return False
        '''

        return not _FallbackPolicy.is_disabled(self) and policy in self

    def is_disabled(self):
        '''Check whether `_FallbackPolicy.FALLBACK_DEVICE is set on the `_FallbackPolicy instance'''

        return _FallbackPolicy.FALLBACK_DISABLE in self


class FallbackBaseException(Exception):
    pass


class ORTModuleDeviceException(FallbackBaseException):
    pass


class ORTModuleTypeError(FallbackBaseException):
    pass


class _FallbackManager(object):
    def __init__(self,
                 policy: _FallbackPolicy,
                 log_level: _logger.LogLevel):
        super().__init__()
        self._log_level = log_level
        self._policy_exception_map = {_FallbackPolicy.FALLBACK_FORCE_TORCH_FORWARD.value: set([ORTModuleDeviceException, ORTModuleTypeError]),
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_DEVICE.value: set([ORTModuleDeviceException]),
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_INPUT.value: set([ORTModuleTypeError]),
                                      _FallbackPolicy.FALLBACK_UNSUPPORTED_OUTPUT.value: set([ORTModuleTypeError])
                                     }
        self._policy = policy
        self._exception = None

    def _handle_exception(self,
                          exception: Exception,
                          policy: Optional[_FallbackPolicy] = None) -> None:

        def _set_exception(policy, exception):


            if policy is not _FallbackPolicy.FALLBACK_DISABLE and \
                    self._policy.is_set(policy) and \
                    (policy.value in self._policy_exception_map and type(exception) in self._policy_exception_map[policy.value]):

                if self._log_level <= _logger.LogLevel.WARNING:
                    warnings.warn(f'Fallback for policy {policy.name} is now pending.', UserWarning)
                self._exception = exception

        if policy is None:
            for policy in _FallbackPolicy:
                _set_exception(policy, exception)
        else:
            _set_exception(policy, exception)

        if self._exception is None:
            raise exception

    def _is_pending(self) -> bool:
        return self._exception is not None

    def _fallback(self, model: torch.nn.Module, *inputs, **kwargs):
        if self._log_level <= _logger.LogLevel.WARNING:
            warnings.warn(f'Fallback due to exception {type(self._exception)} was triggered.', UserWarning)
        return model(*inputs, **kwargs)
