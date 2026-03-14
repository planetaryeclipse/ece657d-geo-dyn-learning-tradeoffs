import torch

from abc import ABC, abstractmethod
from typing import List


class ManifoldCoordSystem(ABC):
    def __init__(self, n: int, ambient_n: int):
        self._n = n
        self._ambient_n = ambient_n

    @property
    def n(self) -> int:
        return self._n

    @property
    def ambient_n(self) -> int:
        return self._ambient_n

    @property
    @abstractmethod
    def default_chart(self) -> str:
        pass

    @property
    @abstractmethod
    def charts(self) -> List[str]:
        pass

    # methods for converting coordinates

    @abstractmethod
    def to_intrinsic(self, chart: str, extrinsic: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def to_extrinsic(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_intrinsic(self, current_chart: str, current_intrinsic: torch.Tensor,
                            target_chart: str) -> torch.Tensor:
        pass

    def endomorphism(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        return self.to_intrinsic(chart, self.to_extrinsic(chart, intrinsic))

    # methods for converting quantities in the tangent space

    @abstractmethod
    def to_intrinsic_ts(self, chart: str, extrinsic: torch.Tensor, extrinsic_ts: torch.Tensor) -> torch.Tensor:
        pass

    def to_extrinsic_ts(self, chart: str, intrinsic: torch.Tensor, intrinsic_ts: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transform_intrinsic_ts(self, current_chart: str, current_intrinsic: torch.Tensor,
                               current_intrinsic_ts: torch.Tensor, target_chart: str) -> torch.Tensor:
        pass

    # @abstractmethod
    # def project_extrinsic_onto_ts(self, extrinsic_vec: torch.Tensor, extrinsic: torch.Tensor) -> torch.Tensor:
    #     pass

    @abstractmethod
    def distance(self, chart: str, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def log(self, chart: str, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def transport_from_q(self, chart: str, p_intrinsic: torch.Tensor, q_intrinsic: torch.Tensor,
                         v_q: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def intrinsic_weights(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        pass

    # quantities defined using the tangent space (note that the metric is defined as a section on the tangent bundle
    # and the Levi-Civita connection is defined with the metric which gives rise to the Christoffel symbols)

    @abstractmethod
    def metric(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def christoffels(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        pass

    # batched coordinate transformations (for convenience during trajectory generation)

    def to_intrinsic_batch(self, chart: str, extrinsic_batch: torch.Tensor):
        num_in_batch = extrinsic_batch.shape[0]
        intrinsic_batch = torch.zeros((num_in_batch, self._n))
        for i in range(num_in_batch):
            intrinsic_batch[i, :] = self.to_intrinsic(chart, extrinsic_batch[i, :])
        return intrinsic_batch

    def to_extrinsic_batch(self, chart: str, intrinsic_batch: torch.Tensor) -> torch.Tensor:
        num_in_batch = intrinsic_batch.shape[0]
        extrinsic_batch = torch.zeros((num_in_batch, self._ambient_n))
        for i in range(num_in_batch):
            extrinsic_batch[i, :] = self.to_extrinsic(chart, intrinsic_batch[i, :])
        return extrinsic_batch

    def transform_intrinsic_batch(self, current_chart: str, current_intrinsic_batch: torch.Tensor,
                                  target_chart: str) -> torch.Tensor:
        num_in_batch = current_intrinsic_batch.shape[0]
        transformed_intrinsic_batch = torch.zeros((num_in_batch, self._n))
        for i in range(num_in_batch):
            transformed_intrinsic_batch[i, :] = self.transform_intrinsic(current_chart, current_intrinsic_batch[i, :],
                                                                         target_chart)
        return transformed_intrinsic_batch

    def to_intrinsic_ts_batch(self, chart: str, extrinsic_batch: torch.Tensor,
                              extrinsic_ts_batch: torch.Tensor) -> torch.Tensor:
        num_in_batch = extrinsic_batch.shape[0]
        intrinsic_ts_batch = torch.zeros((num_in_batch, self._n))
        for i in range(num_in_batch):
            intrinsic_ts_batch[i, :] = self.to_intrinsic_ts(chart, extrinsic_batch[i, :], extrinsic_ts_batch[i, :])
        return intrinsic_ts_batch

    def to_extrinsic_ts_batch(self, chart: str, intrinsic_batch: torch.Tensor,
                              intrinsic_ts_batch: torch.Tensor) -> torch.Tensor:

        num_in_batch = intrinsic_batch.shape[0]
        extrinsic_ts_batch = torch.zeros((num_in_batch, self._ambient_n))
        for i in range(num_in_batch):
            extrinsic_ts_batch[i, :] = self.to_extrinsic_ts(chart, intrinsic_batch[i, :], intrinsic_ts_batch[i, :])
        return extrinsic_ts_batch

    def transform_intrinsic_ts_batch(self, current_chart: str, current_intrinsic_batch: torch.Tensor,
                                     current_intrinsic_ts_batch: torch.Tensor, target_chart: str) -> torch.Tensor:
        num_in_batch = current_intrinsic_batch.shape[0]
        transformed_intrinsic_ts_batch = torch.zeros((num_in_batch, self._n))
        for i in range(num_in_batch):
            transformed_intrinsic_ts_batch[i, :] = self.transform_intrinsic_ts_batch(current_chart,
                                                                                     current_intrinsic_batch[i, :],
                                                                                     current_intrinsic_ts_batch[i, :],
                                                                                     target_chart)
        return transformed_intrinsic_ts_batch

    def intrinsic_weights_batch(self, chart: str, intrinsic_batch: torch.Tensor) -> torch.Tensor:
        num_in_batch = intrinsic_batch.shape[0]
        weights_batch = torch.zeros((num_in_batch,))
        for i in range(num_in_batch):
            weights_batch[i] = self.intrinsic_weights(chart, intrinsic_batch[i, :])
        return weights_batch

    def endomorphism_batch(self, chart: str, intrinsic_batch: torch.Tensor) -> torch.Tensor:
        num_in_batch = intrinsic_batch.shape[0]
        endomorphism_batch = torch.zeros((num_in_batch, self._n))
        for i in range(num_in_batch):
            endomorphism_batch[i, :] = self.to_extrinsic(chart, intrinsic_batch[i, :])
        return endomorphism_batch
