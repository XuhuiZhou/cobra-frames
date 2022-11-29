from __future__ import annotations

from typing import Any


class BaseSBFModel(object):
    def train(self, *args, **kwargs) -> BaseSBFModel:
        raise NotImplementedError

    def predict(self, *args, **kwargs) -> Any:
        raise NotImplementedError
