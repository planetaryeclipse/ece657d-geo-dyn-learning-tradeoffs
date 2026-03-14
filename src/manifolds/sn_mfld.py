import torch
import itertools

from src.manifolds.coord_sys import ManifoldCoordSystem

from typing import List
from torch.autograd.functional import jacobian


# Sn manifold (n-dimensional hypersphere smoothly embedded in Rn+1)

def to_intrinsic(euclid: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    euclid_n = euclid.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1
    intrinsic = torch.zeros((n,))

    if n == 1:
        intrinsic[0] = torch.atan2(euclid[1] / radius, euclid[0] / radius)
    else:
        intrinsic[0] = torch.acos(euclid[0] / radius)
        cum_prod = torch.sin(intrinsic[0])

        for i in range(1, n - 1):
            intrinsic[i] = torch.acos(euclid[i] / radius / cum_prod)
            cum_prod = cum_prod * torch.sin(intrinsic[i])  # re-assigned to prevent autograd error when differentiating
        intrinsic[-1] = torch.atan2(euclid[-2] / radius / cum_prod, euclid[-1] / radius / cum_prod)
    return intrinsic


def to_extrinsic(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    n = intrinsic.shape[0]
    euclid = torch.zeros((n + 1,))

    if n == 1:  # implying extrinsic is at least dim 2
        euclid[0] = radius * torch.cos(intrinsic[0])
        euclid[1] = radius * torch.sin(intrinsic[0])
    else:  # implying extrinsic is at least dim 3
        euclid[0] = radius * torch.cos(intrinsic[0])
        cum_prod = torch.sin(intrinsic[0])
        for i in range(1, n - 1):
            euclid[i] = radius * torch.cos(intrinsic[i]) * cum_prod  # re-assigned
            cum_prod = cum_prod * torch.sin(intrinsic[i])
        euclid[-2] = radius * torch.sin(intrinsic[-1]) * cum_prod
        euclid[-1] = radius * torch.cos(intrinsic[-1]) * cum_prod

    return euclid


def _intrinsic_ts_basis_in_euclid(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    # the basis for the local tangent space is the jacobian matrix of the immersion into the ambient space (note that
    # in this case the basis vectors are the column vectors of the resulting jacobian matrix)
    coord_jacs = jacobian(lambda p: to_extrinsic(p, radius), intrinsic, create_graph=True)
    return coord_jacs
    # return torch.cumsum(coord_jacs, dim=0)


def to_intrinsic_ts(euclid: torch.Tensor, euclid_ts: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    euclid_n = euclid.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1

    # finds the local basis of the tangent space
    intrinsic = to_intrinsic(euclid, radius)
    ts_basis_in_extrinsic = _intrinsic_ts_basis_in_euclid(intrinsic, radius)

    # projects the Euclidean vector onto the basis of the tangent space
    vec_dot_with_basis = torch.tensordot(euclid_ts, ts_basis_in_extrinsic, dims=([0], [0]))
    basis_dot = torch.diag(torch.tensordot(
        ts_basis_in_extrinsic, ts_basis_in_extrinsic, dims=([0], [0])))

    intrinsic_ts = vec_dot_with_basis / basis_dot

    # NOTE: not checking that this vector actually is on the tangent space, the purpose of this project is not to be
    # a fully-fledged differential geometry library

    return intrinsic_ts


def to_extrinsic_ts(intrinsic: torch.Tensor, intrinsic_ts: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    ts_basis_in_extrinsic = _intrinsic_ts_basis_in_euclid(intrinsic, radius)

    # scales the basis vectors (columns of the extrinsic basis) by the intrinsic coordinates
    scaled_basis = torch.tensordot(torch.diag(intrinsic_ts), ts_basis_in_extrinsic, dims=([1], [1]))
    extrinsic_vec = torch.sum(scaled_basis, dim=0)

    return extrinsic_vec


def project_extrinsic_vec_onto_ts(extrinsic_vec: torch.Tensor, extrinsic: torch.Tensor,
                                  radius: float = 1.0) -> torch.Tensor:
    # note code above has been duplicated but it serves different purposes (and in a fully-fledged libraries would have
    # different respective checks) so this is left separate

    # finds the local basis of the tangent space
    intrinsic = to_intrinsic(extrinsic, radius)
    ts_basis_in_extrinsic = _intrinsic_ts_basis_in_euclid(intrinsic, radius)

    # projects the Euclidean vector onto the basis of the tangent space
    dot_extrinsic_with_basis = torch.tensordot(extrinsic_vec, ts_basis_in_extrinsic, dims=([0], [0]))
    dot_basis = torch.diag(torch.tensordot(ts_basis_in_extrinsic, ts_basis_in_extrinsic, dims=([0], [0])))

    factor_on_basis = dot_extrinsic_with_basis / dot_basis
    scaled_basis = torch.tensordot(factor_on_basis, ts_basis_in_extrinsic, dims=([0], [1]))

    return scaled_basis


def _switch_antipodal_coords(coords: torch.Tensor, switch_coords: List[bool]) -> torch.Tensor:
    continuous_coords = (coords + 2 * torch.pi) - torch.Tensor([torch.pi if switch else 0 for switch in switch_coords])
    switched_coords = torch.tensor(
        [coord if coord <= torch.pi else -(2 * torch.pi - coord) for coord in continuous_coords])

    return switched_coords


def to_other_intrinsic(intrinsic: torch.Tensor) -> torch.Tensor:
    n = intrinsic.shape[0]
    total_charts = 2 ** intrinsic.shape[0]  # antipodal chart for each coordinate

    intrinsic_charts = torch.zeros((total_charts, n))
    for i, antipodal in enumerate(itertools.product([False, True], repeat=n)):
        print(f"intrinsic: {intrinsic}")
        assert False
        intrinsic_charts[i, :] = _switch_antipodal_coords(intrinsic, antipodal)

    return intrinsic_charts


def metric(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    coord_jacs = jacobian(lambda p: to_extrinsic(p, radius), intrinsic, create_graph=True)
    g = torch.tensordot(coord_jacs, coord_jacs, dims=([0], [0]))
    return g


def christoffels(intrinsic: torch.Tensor, radius: float = 1.0) -> torch.Tensor:
    g = metric(intrinsic, radius)
    g_partials = jacobian(lambda p: metric(p, radius), intrinsic,
                          create_graph=True)  # adds index at end due to partials

    # computes the connection coefficients of the Levi-Civita connection using the metric thereby describing the
    # curvature of the n-dimensional hypersphere in the intrinsic coordinate system
    conn_coeffs = 0.5 * torch.tensordot(g.inverse(), g_partials + torch.transpose(g_partials, 1, 2) - torch.transpose(
        torch.transpose(g_partials, 1, 2), 0, 1), dims=([1], [0]))

    return conn_coeffs


def _generate_antipodal_switch(n: int, antipodal_idx: int) -> List[bool]:
    # unlike earlier where we used the cartesian product as iteration over all the antipodal points, we use a more
    # efficient method to prevent generating a large list unnecessarily and rather just treat the number as binary
    # where a value of 1 indicates using the antipodal coord for that chart

    switch_coords = [(antipodal_idx << i) & 1 == 1 for i in range(n)]
    return switch_coords


class HypersphereManifold(ManifoldCoordSystem):
    def __init__(self, n: int, radius: float = 1.0):
        super().__init__(n, n + 1)

        self._radius = radius

        num_charts = 2 ** n  # due to the antipodal points
        self._chart_labels = [f"U{i}" for i in range(num_charts)]
        self._chart_nums = {label: i for i, label in enumerate(self._chart_labels)}

    @property
    def radius(self):
        return self._radius

    @property
    def default_chart(self) -> str:
        return "U0"

    @property
    def charts(self) -> List[str]:
        return self._chart_labels

    def to_intrinsic(self, chart: str, extrinsic: torch.Tensor) -> torch.Tensor:
        default_intrinsic = to_intrinsic(extrinsic, self._radius)
        intrinsic = self.transform_intrinsic(self.default_chart, default_intrinsic, chart)
        return intrinsic

    def to_extrinsic(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        default_intrinsic = self.transform_intrinsic(chart, intrinsic, self.default_chart)
        extrinsic = to_extrinsic(default_intrinsic, self._radius)
        return extrinsic

    def transform_intrinsic(self, current_chart: str, current_intrinsic: torch.Tensor,
                            target_chart: str) -> torch.Tensor:
        current_antipodal_switch = _generate_antipodal_switch(self.n, self._chart_nums[current_chart])
        target_antipodal_switch = _generate_antipodal_switch(self.n, self._chart_nums[target_chart])

        transform_switch = [current != target for current, target in
                            zip(current_antipodal_switch, target_antipodal_switch)]

        return _switch_antipodal_coords(current_intrinsic, transform_switch)

    def to_intrinsic_ts(self, chart: str, extrinsic: torch.Tensor, extrinsic_ts: torch.Tensor) -> torch.Tensor:
        # for this hypersphere manifold even though we have shifted the positions between the various charts we have not
        # changed the orientation so the tangent spaces remain aligned
        intrinsic_ts = to_intrinsic_ts(extrinsic, extrinsic_ts, self._radius)
        return intrinsic_ts

    def to_extrinsic_ts(self, chart: str, intrinsic: torch.Tensor, intrinsic_ts: torch.Tensor) -> torch.Tensor:
        extrinsic_ts = to_extrinsic_ts(intrinsic, intrinsic_ts, self._radius)
        return extrinsic_ts

    def transform_intrinsic_ts(self, current_chart: str, current_intrinsic: torch.Tensor,
                               current_intrinsic_ts: torch.Tensor, target_chart: str) -> torch.Tensor:
        return current_intrinsic_ts

    # 
    # def project_extrinsic_onto_ts(self, extrinsic_vec: torch.Tensor, extrinsic: torch.Tensor):
    #     return project_extrinsic_vec_onto_ts(extrinsic_vec, extrinsic, self._radius)

    def distance(self, chart: str, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_extrinsic = self.to_extrinsic(chart, p)
        q_extrinsic = self.to_extrinsic(chart, q)

        # computes the distance by first computing the angle between the points in the intrinsic space then computes the
        # distance by evaluating the arc length of the hypersphere
        ang = torch.arccos(torch.dot(p_extrinsic, q_extrinsic) / self._radius ** 2)
        arc_len = self._radius * ang

        return arc_len

    def log(self, chart: str, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_extrinsic = self.to_extrinsic(chart, p)
        q_extrinsic = self.to_extrinsic(chart, q)

        d_extrinsic = q_extrinsic - p_extrinsic
        d_proj_ts_at_p_extrinsic = project_extrinsic_vec_onto_ts(d_extrinsic, p_extrinsic, self._radius)

        v_ts_at_p_extrinsic = (
                d_proj_ts_at_p_extrinsic / torch.linalg.norm(d_proj_ts_at_p_extrinsic) * self.distance(chart, p, q))

        return self.to_intrinsic_ts(chart, p_extrinsic, v_ts_at_p_extrinsic)

    def transport_from_q(self, chart: str, p_intrinsic: torch.Tensor, q_intrinsic: torch.Tensor,
                         v_q: torch.Tensor) -> torch.Tensor:
        p_extrinsic = self.to_extrinsic(chart, p_intrinsic)
        q_extrinsic = self.to_extrinsic(chart, q_intrinsic)
        v_q_extrinsic = self.to_extrinsic_ts(chart, q_intrinsic, v_q)

        d_extrinsic = q_extrinsic - p_extrinsic

        norm_d_proj_ts_at_p_extrinsic = project_extrinsic_vec_onto_ts(d_extrinsic, p_extrinsic, self._radius)
        norm_d_proj_ts_at_p_extrinsic /= torch.linalg.norm(norm_d_proj_ts_at_p_extrinsic)

        norm_d_proj_ts_at_q_extrinsic = project_extrinsic_vec_onto_ts(d_extrinsic, q_extrinsic, self._radius)
        norm_d_proj_ts_at_q_extrinsic /= torch.linalg.norm(norm_d_proj_ts_at_q_extrinsic)

        v_q_parallel = torch.dot(v_q_extrinsic, norm_d_proj_ts_at_q_extrinsic) * norm_d_proj_ts_at_q_extrinsic
        v_q_perp = v_q_extrinsic - v_q_parallel
        v_p_extrinsic = v_q_perp + torch.dot(v_q_extrinsic,
                                             norm_d_proj_ts_at_q_extrinsic) * norm_d_proj_ts_at_p_extrinsic

        return self.to_intrinsic_ts(chart, p_extrinsic, v_p_extrinsic)

    def intrinsic_weights(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        # for this manifold the chart does not affect the weighting as there are an equal balance of all the charts so
        # we just need to return the scaled distance from the antipodal point (measured in each chart which is the point
        # where the coordinate crossover occurs)
        n = intrinsic.shape[0]
        return torch.sum(1.0 - torch.abs(intrinsic) / torch.pi) / n

    def metric(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        default_intrinsic = self.transform_intrinsic(chart, intrinsic, self.default_chart)
        return metric(default_intrinsic, self._radius)

    def christoffels(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        default_intrinsic = self.transform_intrinsic(chart, intrinsic, self.default_chart)
        return christoffels(default_intrinsic, self._radius)
