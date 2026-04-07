import torch
import torch.nn as nn

import itertools

from typing import Type, Tuple, Optional, Iterator


class DynamicsModel(nn.Module):
    def __init__(self,
                 num_input_features: int,
                 num_output_features: int,
                 num_hidden_layers: int,
                 num_nodes_in_hidden_layer: int,
                 activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()

        self._num_input_features = num_input_features
        self._num_output_features = num_output_features

        # allows parameterizing the network so we can test across different multiple model parameters to ensure that the
        # trend between provided data and intrinsic dimension holds across different model complexities
        self._num_hidden_layers = num_hidden_layers
        self._num_nodes_in_hidden_layer = num_nodes_in_hidden_layer
        self._activation_fn = activation_fn

        # input layer is consistent across all model parameterizations
        layers = [nn.Linear(num_input_features, num_nodes_in_hidden_layer), activation_fn()]

        # constructs the remaining model structure
        for i in range(num_hidden_layers):
            layers.append(nn.Linear(num_nodes_in_hidden_layer, num_nodes_in_hidden_layer))
            layers.append(activation_fn())
        layers.append(nn.Linear(num_nodes_in_hidden_layer, num_output_features))
        self._sequential = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self._sequential(x)


# implements the decoder/encoder scheme for learning dynamics using the embedding

class CoordDecoder(nn.Module):
    def __init__(self,
                 intrin_dim: int, extrin_dim: int, num_hidden_layers: int, num_nodes_per_hidden_layer: int,
                 activation_fn: Type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self._intrin_dim = intrin_dim
        self._extrin_dim = extrin_dim
        self._num_hidden_layers = num_hidden_layers
        self._num_nodes_per_hidden_layer = num_nodes_per_hidden_layer

        # print(f"intrin_dim: {intrin_dim}")
        # print(f"extrin_dim: {extrin_dim}")

        layers = [nn.Linear(2 * intrin_dim, num_nodes_per_hidden_layer), activation_fn()]
        layers.extend([nn.Linear(num_nodes_per_hidden_layer, num_nodes_per_hidden_layer),
                       activation_fn()] * num_hidden_layers)
        layers.extend([nn.Linear(num_nodes_per_hidden_layer, 2 * extrin_dim), activation_fn()])
        self._sequential = nn.Sequential(*layers)

    @property
    def intrin_dim(self) -> int:
        return self._intrin_dim

    @property
    def extrin_dim(self) -> int:
        return self._extrin_dim

    def forward(self, p, v) -> Tuple[torch.Tensor, torch.Tensor]:
        if p.dim() == 1:
            p = torch.unsqueeze(p, dim=0)
            v = torch.unsqueeze(v, dim=0)

        # print(f"p: {p.shape}, device={p.device}")
        # print(f"v: {v.shape}, device={v.device}")
        extrinsic = self._sequential(torch.concat([p, v], dim=1))
        extrin_p, extrin_v = extrinsic[:, :self._extrin_dim], extrinsic[:, self._extrin_dim:]

        return extrin_p.squeeze(0), extrin_v.squeeze(0)


# note that the velocity is not needed to reconstruct the intrinsic position but we do require the extrinsic position
# and velocity if we want the intrinsic velocity so we construct two separate encoders (for our training runs with the
# disposable velocity encoder)

class CoordPosEncoder(nn.Module):
    def __init__(self,
                 intrin_dim: int, extrin_dim: int, num_hidden_layers: int, num_nodes_per_hidden_layer: int,
                 activation_fn: Type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self._intrin_dim = intrin_dim
        self._extrin_dim = extrin_dim
        self._num_hidden_layers = num_hidden_layers
        self._num_nodes_per_hidden_layer = num_nodes_per_hidden_layer

        layers = [nn.Linear(extrin_dim, num_nodes_per_hidden_layer), activation_fn()]
        layers.extend([nn.Linear(num_nodes_per_hidden_layer, num_nodes_per_hidden_layer),
                       activation_fn()] * num_hidden_layers)
        layers.extend([nn.Linear(num_nodes_per_hidden_layer, intrin_dim), activation_fn()])
        self._sequential = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self._sequential(x)


class CoordVelEncoder(nn.Module):
    def __init__(self,
                 intrin_dim: int, extrin_dim: int, num_hidden_layers: int, num_nodes_per_hidden_layer: int,
                 activation_fn: Type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self._intrin_dim = intrin_dim
        self._extrin_dim = extrin_dim
        self._num_hidden_layers = num_hidden_layers
        self._num_nodes_per_hidden_layer = num_nodes_per_hidden_layer

        layers = [nn.Linear(2 * extrin_dim, num_nodes_per_hidden_layer), activation_fn()]
        layers.extend([nn.Linear(num_nodes_per_hidden_layer, num_nodes_per_hidden_layer),
                       activation_fn()] * num_hidden_layers)
        layers.extend([nn.Linear(num_nodes_per_hidden_layer, intrin_dim), activation_fn()])
        self._sequential = nn.Sequential(*layers)

    def forward(self, p, v) -> torch.Tensor:
        return self._sequential(torch.concat([p, v], dim=1))


class EmbeddedDynamicsModel(nn.Module):
    def __init__(self,
                 intrin_dim: int, learn_extrin_dim: int,
                 enc_num_layers: int, enc_num_nodes_per_layer: int,
                 dec_num_layers: int, dec_num_nodes_per_layer: int,
                 mlp_num_layers: int, mlp_num_nodes_per_layer: int, with_prev_state: bool = False,
                 activation_fn: Type[nn.Module] = nn.LeakyReLU):
        super().__init__()
        self._intrin_dim = intrin_dim
        self._learn_extrin_dim = learn_extrin_dim
        self._enc_num_layers = enc_num_layers
        self._enc_num_nodes_per_layer = enc_num_nodes_per_layer
        self._dec_num_layers = dec_num_layers
        self._dec_num_nodes_per_layer = dec_num_nodes_per_layer
        self._mlp_num_layers = mlp_num_layers
        self._mlp_num_nodes_per_layer = mlp_num_nodes_per_layer
        self._with_prev_state = with_prev_state  # basically if we're passing in a previous

        self._decoder = CoordDecoder(  # decodes both position and velocity
            intrin_dim, learn_extrin_dim,
            dec_num_layers, dec_num_nodes_per_layer, )

        self._pos_encoder = CoordPosEncoder(  # encodes only position
            intrin_dim, learn_extrin_dim, enc_num_layers,
            enc_num_nodes_per_layer, )
        self._vel_encoder = CoordVelEncoder(  # encodes only velocity
            intrin_dim, learn_extrin_dim, enc_num_layers,
            enc_num_nodes_per_layer, )

        # sets up the prediction layer which predicts the next state from previous state(s) in extrinsic coordinates
        layers = [
            nn.Linear(4 * learn_extrin_dim if with_prev_state else 2 * learn_extrin_dim, mlp_num_nodes_per_layer),
            activation_fn(),
        ]
        layers.extend([nn.Linear(mlp_num_nodes_per_layer, mlp_num_nodes_per_layer),
                       activation_fn()] * mlp_num_layers)
        layers.extend([nn.Linear(mlp_num_nodes_per_layer, 1 * learn_extrin_dim), activation_fn()])
        self._mlp_sequential = nn.Sequential(*layers)

    @property
    def with_prev_state(self) -> bool:
        return self._with_prev_state

    def forward(self, p: torch.Tensor, v: torch.Tensor, prev_p: Optional[torch.Tensor] = None,
                prev_v: Optional[torch.Tensor] = None) -> torch.Tensor:
        # predicts the updated position on the hypersphere (in intrinsic coordinates)
        # NOTE: utilize the other methods to get other quantities for the various embedded training schemes
        # NOTE: if prev_p and prev_v are provided then we also need to run these through the decoder separately

        extrin_p, extrin_v = self._decoder(p, v)
        if prev_p is None:
            next_extrin_p = self._mlp_sequential(torch.concat([extrin_p, extrin_v], dim=1))
        else:
            extrin_prev_p, extrin_prev_v = self._decoder(prev_p, prev_v)
            next_extrin_p = self._mlp_sequential(
                torch.concat([extrin_p, extrin_v, extrin_prev_p, extrin_prev_v], dim=1))

        next_intrin_p = self._pos_encoder(next_extrin_p)
        return next_intrin_p

    def get_extrinsic_from_decoder(self, p, v) -> Tuple[torch.Tensor, torch.Tensor]:
        # runs the intrinsic pos/vel through the decoder to get extrinsic pos/vel
        return self._decoder(p, v)

    def get_intrinsic_pos_from_encoder(self, p_extrin) -> torch.Tensor:
        # runs the extrinsic pos/vel through the encoder to get intrinsic pos
        return self._pos_encoder(p_extrin)

    def get_intrinsic_vel_from_encoder(self, p_extrin, v_extrin) -> torch.Tensor:
        # runs the extrinsic pos/vel through the encoder to get intrinsic vel
        return self._vel_encoder(p_extrin, v_extrin)

    def get_intrinsic_pos_from_dec_enc(self, p, v) -> torch.Tensor:
        # runs the intrinsic pos/vel through the decoder and re-encodes to get the pos
        extrin_p, _ = self._decoder(p, v)
        return self._pos_encoder(extrin_p)

    def get_intrinsic_pos_vel_from_dec_enc(self, p, v) -> Tuple[torch.Tensor, torch.Tensor]:
        # runs the intrinsic pos/vel throught the decoder and re-encodes to get the vel
        extrin_p, extrin_v = self._decoder(p, v)

        intrin_p = self._pos_encoder(extrin_p)
        intrin_v = self._vel_encoder(extrin_p, extrin_v) # velocity encoder not used in regular forward path

        return intrin_p, intrin_v

    def get_dyn_mlp_params(self) -> Iterator[nn.Parameter]:
        return self._mlp_sequential.parameters()

    def get_decoder(self) -> CoordDecoder:
        return self._decoder

    def get_pos_encoder(self) -> CoordPosEncoder:
        return self._pos_encoder

