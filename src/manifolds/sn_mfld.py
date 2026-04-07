import torch
import itertools

import math

from src.manifolds.coord_sys import ManifoldCoordSystem

from typing import List, Tuple, Optional
from torch.autograd.functional import jacobian

SINGULARITY_CORR_EPS = 1E-10
ZERO_NORM_EPS = 1E-6


def _axis_permute(permute_idx: int, n: int) -> Tuple[int, ...]:
    if permute_idx < 0 or permute_idx >= math.factorial(n):
        raise ValueError("Permutation index out of range")

    result = []
    pending_idxs = list(range(n))
    temp_permute_idx = permute_idx

    for i in range(n, 0, -1):
        fact = math.factorial(i - 1)
        idx, temp_permute_idx = divmod(temp_permute_idx, fact)
        result.append(pending_idxs.pop(idx))

    return tuple(result)


def _permute_idx_from_permutation(permutation: Tuple[int, ...], n: int) -> int:
    elements = list(range(n))
    idx = 0

    for i, p in enumerate(permutation):
        pos = elements.index(p)
        idx += pos * math.factorial(n - i - 1)
        elements.pop(pos)

    return idx


def _non_singular_chart_id(extrinsic: torch.Tensor) -> int:
    extrinsic_n = extrinsic.shape[0]
    n = extrinsic_n - 1

    # NOTE: there's probably a more elegant way of doing this but for now I just need this to work so brute force it is

    # print(f"extrinsic coords: {extrinsic}")
    # print("finding nonsingular chart id...")

    cum_sqr_dist_from_pi_2_dists = []
    for i in range(math.factorial(extrinsic_n)):
        # print(f"i: {i}")
        intrinsic_coords = to_intrinsic(extrinsic, i)
        # print(f"chart_idx: {i}, intrinsic_coords: {intrinsic_coords}")

        # the singularity only occurs in the first n-1 coordinates in the intrinsic coordinates which if the value is
        # either 0 or pi then i + 1, and remaining, extrinsic coordinates collapse to 0, so we need to choose the set
        # of coords that minimizes distance to all the pi/2 coords

        phi_coords = intrinsic_coords[:-1] if n > 1 else torch.tensor([])  # if S1 then no phi angles
        theta_coord = intrinsic_coords[-1]

        # print(f"phi_coords: {phi_coords}")
        # print(f"theta_coord: {theta_coord}")

        # note the relative scaling here by half the domain of the phis and theta so the theta selection does not
        # overpower selection of the phi angles when choosing a point away from singularity
        phi_dists = torch.abs((torch.pi / 2 * torch.ones_like(phi_coords) - phi_coords)) / (torch.pi / 2)
        theta_dist = torch.abs(torch.pi - theta_coord) / torch.pi

        # print(f"phi_dists: {phi_dists}")
        # print(f"theta_dist: {theta_dist}")

        cum_dist_from_pi_2 = torch.sum(phi_dists) + theta_dist
        # print(f"cum distance from pi_2: {cum_dist_from_pi_2}")
        cum_sqr_dist_from_pi_2_dists.append(cum_dist_from_pi_2)

    return int(torch.argmin(torch.tensor(cum_sqr_dist_from_pi_2_dists), dim=0))


# def _non_singular_chart_id(extrinsic: torch.Tensor) -> int:
#     extrinsic_n = extrinsic.shape[0]
#     abs_components = list(torch.abs(extrinsic))
#
#     # first axis is chosen with minimum absolute component value so it places the first component evaluated by cos away
#     # from the singularities at -1 and 1 (and hence midway would be 0)
#     min_component_idxs = sorted(range(extrinsic_n), key=lambda idx: abs_components[idx], reverse=False)
#     first_idx = min_component_idxs.pop(0)
#
#     # chooses the remaining axes by maximizing the absolute component values as successive components when producing the
#     # extrinsic representation require multiplication by the sin of the angle which breaks down at 0 and pi so we choose
#     # the maximum so that way we remain centered around pi/2 or -pi/2
#
#     remaining_abs_components = [abs_components[idx] for idx in min_component_idxs]
#     print(f"remaining_abs_components: {remaining_abs_components}")
#     print(f"min_component_idxs: {min_component_idxs}")
#     max_component_idxs = sorted(range(len(min_component_idxs)), key=lambda idx: remaining_abs_components[idx],
#                                 reverse=True)
#     max_component_idxs = [min_component_idxs[idx] for idx in max_component_idxs]
#     # max_component_idxs = [idx if idx < first_idx else idx + 1 for idx in
#     #                       max_component_idxs]  # accounts for removing the first component
#
#     ang_idxs = tuple([first_idx, *max_component_idxs])
#     chart_id = _permute_idx_from_permutation(ang_idxs, extrinsic_n)
#     return chart_id


# Sn manifold (n-dimensional hypersphere smoothly embedded in Rn+1)


def to_intrinsic(euclid: torch.Tensor, chart_idx: int) -> torch.Tensor:
    euclid_n = euclid.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1
    intrinsic = torch.zeros((n,), dtype=euclid.dtype)
    euclid_axes = _axis_permute(chart_idx, euclid_n)

    if n == 1:
        intrinsic[0] = torch.atan2(euclid[euclid_axes[1]], euclid[euclid_axes[0]])
    else:
        for i in range(n - 1):
            subnorm = torch.linalg.norm(euclid[list(euclid_axes[(i + 1):])])
            intrinsic[i] = torch.atan2(subnorm, euclid[euclid_axes[i]])
        intrinsic[-1] = torch.atan2(euclid[euclid_axes[-2]], euclid[euclid_axes[-1]])
    intrinsic[-1] = (intrinsic[-1] + 2 * torch.pi) % (2 * torch.pi)

    return intrinsic


def to_extrinsic(intrinsic: torch.Tensor, chart_idx: int, radius: float = 1.0) -> torch.Tensor:
    n = intrinsic.shape[0]
    euclid = torch.zeros((n + 1,), dtype=intrinsic.dtype)

    euclid_axes = _axis_permute(chart_idx, n + 1)

    if n == 1:  # implying extrinsic is at least dim 2
        euclid[euclid_axes[0]] = radius * torch.cos(intrinsic[0])
        euclid[euclid_axes[1]] = radius * torch.sin(intrinsic[0])
    else:  # implying extrinsic is at least dim 3
        euclid[euclid_axes[0]] = radius * torch.cos(intrinsic[0])
        cum_prod = torch.sin(intrinsic[0])
        for i in range(1, n - 1):
            euclid[euclid_axes[i]] = radius * torch.cos(intrinsic[i]) * cum_prod  # re-assigned
            cum_prod = cum_prod * torch.sin(intrinsic[i])
        euclid[euclid_axes[-2]] = radius * torch.sin(intrinsic[-1]) * cum_prod
        euclid[euclid_axes[-1]] = radius * torch.cos(intrinsic[-1]) * cum_prod

    return euclid


# def _complete_ts_manually(degenerate_jacob: torch.Tensor, chart_id: int) -> torch.Tensor:
#     n = degenerate_jacob.shape[1]  # intrinsic dimension
#
#     permuted_axes = _axis_permute(chart_id, n)  # the ordering of columns of jacobian wrt intrinsic coords
#
#     pass


def _intrinsic_ts_basis_in_extrinsic(intrinsic: torch.Tensor, chart_idx: int, radius: float = 1.0) -> torch.Tensor:
    n = intrinsic.shape[0]

    # if the provided coordinates are at a singularity then it finds coordinates in the valid range that are close
    # enough to the singular coordinates but still have full rank when evaluated with the jacobian
    modified_intrinsic_coords = torch.zeros((n,), dtype=intrinsic.dtype)

    modified_intrinsic_coords[:n - 1] = torch.clip(intrinsic[:n - 1], SINGULARITY_CORR_EPS,
                                                   torch.pi - SINGULARITY_CORR_EPS)
    modified_intrinsic_coords[-1] = torch.clip(intrinsic[-1], SINGULARITY_CORR_EPS, 2 * torch.pi - SINGULARITY_CORR_EPS)

    coord_jac = jacobian(lambda p: to_extrinsic(p, chart_idx, radius), modified_intrinsic_coords, create_graph=True)
    return coord_jac


def to_intrinsic_ts(extrinsic: torch.Tensor, extrinsic_ts: torch.Tensor, chart_idx: int,
                    radius: float = 1.0, ignore_basis_check: bool = False) -> torch.Tensor:
    # print("TO_INTRINSIC_TS")

    euclid_n = extrinsic.shape[0]  # dimension of the ambient Euclidean space
    if euclid_n < 2:
        raise ValueError("Euclidean dimension must be >= 2")

    n = euclid_n - 1

    # print("INTRINSIC_TS")
    # print(f"extrinsic: {extrinsic}, extrinsic_ts: {extrinsic_ts}")

    # NOTE: not checking that this vector actually is on the tangent space, the purpose of this project is not to be
    # a fully-fledged differential geometry library

    # finds the local basis of the tangent space (note that this chart will be approximate if at a singular point  of
    # the chart by finding a "close-enough" point in the valid range of the chart)
    intrinsic = to_intrinsic(extrinsic, chart_idx)
    ts_basis_in_extrinsic = _intrinsic_ts_basis_in_extrinsic(intrinsic, chart_idx, radius)

    if not ignore_basis_check:
        _check_ts_basis(chart_idx, ts_basis_in_extrinsic)

    # print(f"intrinsic: {intrinsic}")
    # print(f"ts_basis_in_extrinsic: {ts_basis_in_extrinsic}")

    # projects the Euclidean vector onto the basis of the tangent space
    # print(f"extrinsic_ts_dtype={extrinsic_ts.dtype}, ts_basis_ine_extrinsic={ts_basis_in_extrinsic.dtype}")
    vec_dot_with_basis = torch.tensordot(extrinsic_ts, ts_basis_in_extrinsic, dims=([0], [0]))

    # print(f"vec_dot_with_basis: {vec_dot_with_basis}")

    basis_dot = torch.diag(torch.tensordot(
        ts_basis_in_extrinsic, ts_basis_in_extrinsic, dims=([0], [0])))

    intrinsic_ts = vec_dot_with_basis / basis_dot

    recon_extrinsic_ts = to_extrinsic_ts(intrinsic, intrinsic_ts, chart_idx, radius)
    recon_err = torch.linalg.norm(recon_extrinsic_ts - extrinsic_ts)

    # print("at end of function...")
    # print(f"intrinsic: {intrinsic}, extrinsic: {extrinsic}")
    # print(f"intrinsic_ts: {intrinsic_ts}, extrinsic_ts: {extrinsic_ts}")
    # print(f"recon_extrinsic_ts: {recon_extrinsic_ts}")
    # print(f"recon_err: {recon_err}")
    # print(f"chart_idx: {chart_idx}")

    if recon_err > ZERO_NORM_EPS:
        raise ValueError("Provided extrinsic vector is not within the extrinsic tangent space")

    # print(f"intrinsic_ts: {intrinsic_ts}")

    return intrinsic_ts


def _check_ts_basis(chart_idx: int, ts_basis_in_extrinsic: torch.Tensor):
    # will fail explicitly
    n = ts_basis_in_extrinsic.shape[1]

    rank = torch.linalg.matrix_rank(ts_basis_in_extrinsic, atol=ZERO_NORM_EPS)
    # print(f"rank: {rank}")

    if rank < n:
        raise ValueError(f"Tangent basis is rank-deficient, use a different chart other than id={chart_idx}")


def to_extrinsic_ts(intrinsic: torch.Tensor, intrinsic_ts: torch.Tensor, chart_idx: int,
                    radius: float = 1.0) -> torch.Tensor:
    ts_basis_in_extrinsic = _intrinsic_ts_basis_in_extrinsic(intrinsic, chart_idx, radius)

    # print(f"(in to_extrinsic_ts) ts_basis_in_extrinsic: {ts_basis_in_extrinsic}")

    _check_ts_basis(chart_idx, ts_basis_in_extrinsic)

    # scales the basis vectors (columns of the extrinsic basis) by the intrinsic coordinates
    extrinsic_vec = torch.tensordot(intrinsic_ts, ts_basis_in_extrinsic, dims=([0], [1]))
    return extrinsic_vec


def project_extrinsic_vec_onto_ts(extrinsic_vec: torch.Tensor, chart_idx: int, extrinsic: torch.Tensor,
                                  radius: float = 1.0) -> Optional[torch.Tensor]:
    # note code above has been duplicated but it serves different purposes (and in a fully-fledged libraries would have
    # different respective checks) so this is left separate

    # finds the local basis of the tangent space
    intrinsic = to_intrinsic(extrinsic, chart_idx)
    ts_basis_in_extrinsic = _intrinsic_ts_basis_in_extrinsic(intrinsic, chart_idx, radius)

    _check_ts_basis(chart_idx, ts_basis_in_extrinsic)

    # print(f"ts_basis_in_extrinsic: {ts_basis_in_extrinsic}")

    # projects the Euclidean vector onto the basis of the tangent space
    dot_extrinsic_with_basis = torch.tensordot(extrinsic_vec, ts_basis_in_extrinsic, dims=([0], [0]))
    dot_basis = torch.diag(torch.tensordot(ts_basis_in_extrinsic, ts_basis_in_extrinsic, dims=([0], [0])))

    # print(f"dot_extrinsic_with_basis: {dot_extrinsic_with_basis}")
    # print(f"dot_basis: {dot_basis}")

    factor_on_basis = dot_extrinsic_with_basis / dot_basis
    scaled_basis = torch.tensordot(factor_on_basis, ts_basis_in_extrinsic, dims=([0], [1]))

    # print(f"factor_on_basis: {factor_on_basis}")
    # print(f"scaled_basis: {scaled_basis}")

    return scaled_basis


# def _switch_antipodal_coords(coords: torch.Tensor, switch_coords: List[bool]) -> torch.Tensor:
#     continuous_coords = (coords + 2 * torch.pi) - torch.Tensor([torch.pi if switch else 0 for switch in switch_coords])
#     switched_coords = torch.tensor(
#         [coord if coord <= torch.pi else -(2 * torch.pi - coord) for coord in continuous_coords])
#
#     return switched_coords

def _to_all_intrinsic(extrinsic: torch.Tensor) -> torch.Tensor:
    extrinsic_n = extrinsic.shape[0]
    intrinsic_n = extrinsic_n - 1

    total_charts = math.factorial(extrinsic_n)
    intrinsic_charts = torch.zeros((total_charts, intrinsic_n))

    for i in range(total_charts):
        intrinsic_charts[i, :] = to_intrinsic(extrinsic, i)

    return intrinsic_charts


def _to_all_intrinsic_ts(extrinsic: torch.Tensor, extrinsic_ts: torch.Tensor, radius: float) -> torch.Tensor:
    extrinsic_n = extrinsic.shape[0]
    intrinsic_n = extrinsic_n - 1

    total_charts = math.factorial(extrinsic_n)
    intrinsic_ts_charts = torch.zeros((total_charts, intrinsic_n), dtype=extrinsic.dtype)

    for i in range(total_charts):
        intrinsic_ts_charts[i, :] = to_intrinsic_ts(extrinsic, extrinsic_ts, i, radius)

    return intrinsic_ts_charts


# def to_other_intrinsic(intrinsic: torch.Tensor, chart_id: int) -> torch.Tensor:
#     n = intrinsic.shape[0]
#
#     to_extrinsic_ts()
#
#     total_charts = math.factorial(n+1) # different permutations of axes available
#     intrinsic_charts = torch.zeros((total_charts, n))
#
#     for i in range(total_charts):


# total_charts = 2 ** intrinsic.shape[0]  # antipodal chart for each coordinate
# intrinsic_charts = torch.zeros((total_charts, n))
# for i, antipodal in enumerate(itertools.product([False, True], repeat=n)):
#     print(f"intrinsic: {intrinsic}")
#     assert False
#     intrinsic_charts[i, :] = _switch_antipodal_coords(intrinsic, antipodal)
#
# return intrinsic_charts

def distance(p_intrinsic: torch.Tensor, q_intrinsic, chart_idx: int, radius: float = 1.0) -> float:
    p_extrinsic = to_extrinsic(p_intrinsic, chart_idx, radius)
    q_extrinsic = to_extrinsic(q_intrinsic, chart_idx, radius)

    # print(f"inside distance, p_extrinsic: {p_extrinsic}, q_extrinsic: {q_extrinsic}")

    # computes the distance by first computing the angle between the points in the intrinsic space then computes the
    # distance by evaluating the arc length of the hypersphere
    ang = torch.arccos(torch.dot(p_extrinsic, q_extrinsic) / radius ** 2)
    arc_len = radius * ang

    return arc_len.item()


def metric(intrinsic: torch.Tensor, chart_idx: int, radius: float = 1.0) -> torch.Tensor:
    coord_jacs = jacobian(lambda p: to_extrinsic(p, chart_idx, radius), intrinsic, create_graph=True)
    g = torch.tensordot(coord_jacs, coord_jacs, dims=([0], [0]))
    return g


def christoffels(intrinsic: torch.Tensor, chart_idx: int, radius: float = 1.0) -> torch.Tensor:
    g = metric(intrinsic, chart_idx, radius)
    g_partials = jacobian(lambda p: metric(p, chart_idx, radius), intrinsic,
                          create_graph=True)  # adds index at end due to partials

    # computes the connection coefficients of the Levi-Civita connection using the metric thereby describing the
    # curvature of the n-dimensional hypersphere in the intrinsic coordinate system
    conn_coeffs = 0.5 * torch.tensordot(g.inverse(), g_partials + torch.transpose(g_partials, 1, 2) - torch.transpose(
        torch.transpose(g_partials, 1, 2), 0, 1), dims=([1], [0]))

    return conn_coeffs


# def _generate_antipodal_switch(n: int, antipodal_idx: int) -> List[bool]:
#     # unlike earlier where we used the cartesian product as iteration over all the antipodal points, we use a more
#     # efficient method to prevent generating a large list unnecessarily and rather just treat the number as binary
#     # where a value of 1 indicates using the antipodal coord for that chart
#
#     switch_coords = [(antipodal_idx << i) & 1 == 1 for i in range(n)]
#     return switch_coords


class HypersphereManifold(ManifoldCoordSystem):
    def __init__(self, n: int, radius: float = 1.0):
        super().__init__(n, n + 1)

        self._radius = radius

        num_charts = math.factorial(n + 1)  # permutations in reconstruction into ambient space
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
        return to_intrinsic(extrinsic, self._chart_nums[chart])

        # default_intrinsic = to_intrinsic(extrinsic, self._chart_nums[chart])
        # intrinsic = self.transform_intrinsic(self.default_chart, default_intrinsic, chart)
        # return intrinsic

    def to_extrinsic(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        return to_extrinsic(intrinsic, self._chart_nums[chart], self._radius)

        # default_intrinsic = self.transform_intrinsic(chart, intrinsic, self.default_chart)
        # extrinsic = to_extrinsic(default_intrinsic, self._chart_nums[chart], self._radius)
        # return extrinsic

    # def transform_intrinsic(self, current_chart: str, current_intrinsic: torch.Tensor,
    #                         target_chart: str) -> torch.Tensor:
    #     current_antipodal_switch = _generate_antipodal_switch(self.n, self._chart_nums[current_chart])
    #     target_antipodal_switch = _generate_antipodal_switch(self.n, self._chart_nums[target_chart])
    #
    #     transform_switch = [current != target for current, target in
    #                         zip(current_antipodal_switch, target_antipodal_switch)]
    #
    #     return _switch_antipodal_coords(current_intrinsic, transform_switch)

    def to_intrinsic_ts(self, chart: str, extrinsic: torch.Tensor, extrinsic_ts: torch.Tensor,
                        ignore_basis_check: bool = False) -> torch.Tensor:
        # for this hypersphere manifold even though we have shifted the positions between the various charts we have not
        # changed the orientation so the tangent spaces remain aligned

        # print(f"TO INTRINSIC TS")
        # print(f"extrinsic: {extrinsic}")
        # print(f"extrinsic_ts: {extrinsic_ts}")

        intrinsic_ts = to_intrinsic_ts(extrinsic, extrinsic_ts, self._chart_nums[chart], self._radius,
                                       ignore_basis_check)
        return intrinsic_ts

    def to_extrinsic_ts(self, chart: str, intrinsic: torch.Tensor, intrinsic_ts: torch.Tensor) -> torch.Tensor:
        extrinsic_ts = to_extrinsic_ts(intrinsic, intrinsic_ts, self._chart_nums[chart], self._radius)
        return extrinsic_ts

    # def transform_intrinsic_ts(self, current_chart: str, current_intrinsic: torch.Tensor,
    #                            current_intrinsic_ts: torch.Tensor, target_chart: str) -> torch.Tensor:
    #     return current_intrinsic_ts

    #
    # def project_extrinsic_onto_ts(self, extrinsic_vec: torch.Tensor, extrinsic: torch.Tensor):
    #     return project_extrinsic_vec_onto_ts(extrinsic_vec, extrinsic, self._radius)

    def nonsingular_chart_id(self, extrinsic: torch.Tensor) -> str:
        return self.charts[_non_singular_chart_id(extrinsic)]

    def distance(self, chart: str, p: torch.Tensor, q: torch.Tensor) -> float:
        return distance(p, q, self._chart_nums[chart], self._radius)

    def log(self, chart: str, p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        p_extrinsic = self.to_extrinsic(chart, p)
        q_extrinsic = self.to_extrinsic(chart, q)

        # print(f"p_extrinsic: {p_extrinsic}, q_extrinsic: {q_extrinsic}")

        # manual check given the norm will be 0 in this case which will lead to a nan result
        if torch.allclose(p_extrinsic, q_extrinsic):
            return torch.zeros(p.shape)

        d_extrinsic = q_extrinsic - p_extrinsic

        # print(f"d_extrinsic: {d_extrinsic}")

        # projects onto the local tangent space if it exists, if d is orthogonal then the resulting point is at the
        # opposite side of the sphere so we just choose any vector in the tangent space
        d_proj_ts_at_p_extrinsic = project_extrinsic_vec_onto_ts(d_extrinsic,
                                                                 self._chart_nums[chart],
                                                                 p_extrinsic,
                                                                 self._radius)
        # print(f"proj norm: {torch.linalg.norm(d_proj_ts_at_p_extrinsic)}")
        if torch.linalg.norm(d_proj_ts_at_p_extrinsic) < ZERO_NORM_EPS:
            d_proj_ts_at_p_extrinsic = self.to_extrinsic_ts(chart, p, torch.ones((self.n,)))

        distance_extrinsic = distance(p, q, self._chart_nums[chart], self._radius)
        d_proj_ts_at_p_extrinsic /= torch.linalg.norm(d_proj_ts_at_p_extrinsic)
        d_proj_ts_at_p_extrinsic *= distance_extrinsic

        log = self.to_intrinsic_ts(chart, p_extrinsic, d_proj_ts_at_p_extrinsic)
        return log

    def transport_from_q(self, chart_p: str, p_intrinsic: torch.Tensor, chart_q: str, q_intrinsic: torch.Tensor,
                         v_q: torch.Tensor) -> torch.Tensor:
        p_extrinsic = self.to_extrinsic(chart_p, p_intrinsic)
        q_extrinsic = self.to_extrinsic(chart_q, q_intrinsic)
        v_q_extrinsic = self.to_extrinsic_ts(chart_q, q_intrinsic, v_q)

        d_extrinsic = q_extrinsic - p_extrinsic

        p_chart_idx = self._chart_nums[chart_p]
        q_chart_idx = self._chart_nums[chart_q]

        norm_d_proj_ts_at_p_extrinsic = project_extrinsic_vec_onto_ts(d_extrinsic, p_chart_idx, p_extrinsic,
                                                                      self._radius)
        norm_d_proj_ts_at_p_extrinsic /= torch.linalg.norm(norm_d_proj_ts_at_p_extrinsic)

        norm_d_proj_ts_at_q_extrinsic = project_extrinsic_vec_onto_ts(d_extrinsic, q_chart_idx, q_extrinsic,
                                                                      self._radius)
        norm_d_proj_ts_at_q_extrinsic /= torch.linalg.norm(norm_d_proj_ts_at_q_extrinsic)

        v_q_parallel = torch.dot(v_q_extrinsic, norm_d_proj_ts_at_q_extrinsic) * norm_d_proj_ts_at_q_extrinsic
        v_q_perp = v_q_extrinsic - v_q_parallel
        v_p_extrinsic = v_q_perp + torch.dot(v_q_extrinsic,
                                             norm_d_proj_ts_at_q_extrinsic) * norm_d_proj_ts_at_p_extrinsic

        return self.to_intrinsic_ts(chart_p, p_extrinsic, v_p_extrinsic)

    def intrinsic_weights(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        # for this manifold the chart does not affect the weighting as there are an equal balance of all the charts so
        # we just need to return the scaled distance from the antipodal point (measured in each chart which is the point
        # where the coordinate crossover occurs)
        n = intrinsic.shape[0]
        return torch.sum(1.0 - torch.abs(intrinsic) / torch.pi) / n

    def intrinsic_coords_validity(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        phi_coords = intrinsic[:-1] if self._n > 1 else torch.tensor([])  # if S1 then no phi angles
        theta_coord = intrinsic[-1]

        # note the relative scaling here by half the domain of the phis and theta so the theta selection does not
        # overpower selection of the phi angles when choosing a point away from singularity
        phi_dists = torch.abs((torch.pi / 2 * torch.ones_like(phi_coords) - phi_coords)) / (torch.pi / 2)
        theta_dist = torch.abs(torch.pi - theta_coord) / torch.pi

        if len(phi_dists) > 0:
            return torch.concat((phi_dists, torch.unsqueeze(theta_dist, dim=-1)), 0)
        else:
            return theta_dist

    def metric(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        return metric(intrinsic, self._chart_nums[chart], self._radius)

    def christoffels(self, chart: str, intrinsic: torch.Tensor) -> torch.Tensor:
        return christoffels(intrinsic, self._chart_nums[chart], self._radius)
