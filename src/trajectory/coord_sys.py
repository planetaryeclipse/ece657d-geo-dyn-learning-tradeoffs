import numpy as np

from abc import ABC, abstractmethod
from typing import List


class CoordSystem(ABC):
    @abstractmethod
    def default_chart(self) -> str:
        pass

    @abstractmethod
    def charts(self) -> List[str]:
        pass

    @abstractmethod
    def to_intrinsic(self, chart: str, extrinsic: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def to_extrinsic(self, chart: str, intrinsic: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def intrinsic_weights(self, chart: str, intrinsic: np.ndarray) -> np.ndarray:
        pass

    def endomorphism(self, chart: str, intrinsic: np.ndarray) -> np.ndarray:
        return self.to_intrinsic(chart, self.to_extrinsic(chart, intrinsic))
