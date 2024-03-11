import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from deep_lagrangian_networks.DeLaN_model import LowTri, LagrangianLayer


class ResLayer(nn.Module):
    def __init__(self, input_size, n_dof, activation=True):
        super(ResLayer, self).__init__()
        # Create layer weights and biases:
        self.n_dof = n_dof
        self.activation = activation
        self.weight = nn.Parameter(torch.Tensor(n_dof, input_size))
        self.bias = nn.Parameter(torch.Tensor(n_dof))
        self.g = nn.ReLU()

    def forward(self, x):
        # Apply Affine Transformation:
        a = F.linear(x, self.weight, self.bias)
        out = self.g(a) if self.activation else a
        return out


class DeepResidualLagrangianNetwork(nn.Module):

    def __init__(self, n_dof, **kwargs):
        super(DeepResidualLagrangianNetwork, self).__init__()

        # Read optional arguments:
        self.residual_layers = [8, 4, 1]
        self.n_width = kwargs.get("n_width", 128)
        self.n_hidden = kwargs.get("n_depth", 1)
        self._b0 = kwargs.get("b_init", 0.1)
        self._b0_diag = kwargs.get("b_diag_init", 0.1)

        self._w_init = kwargs.get("w_init", "xavier_normal")
        self._g_hidden = kwargs.get("g_hidden", np.sqrt(2.))
        self._g_output = kwargs.get("g_hidden", 0.125)
        self._p_sparse = kwargs.get("p_sparse", 0.2)
        self._epsilon = kwargs.get("diagonal_epsilon", 1.e-5)

        # Construct Weight Initialization:
        if self._w_init == "xavier_normal":

            # Construct initialization function:
            def init_hidden(layer):

                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.xavier_normal_(layer.weight, output_gain)

        elif self._w_init == "orthogonal":

            # Construct initialization function:
            def init_hidden(layer):
                # Set the Hidden Gain:
                if self._g_hidden <= 0.0: hidden_gain = torch.nn.init.calculate_gain('relu')
                else: hidden_gain = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, hidden_gain)

            def init_output(layer):
                # Set Output Gain:
                if self._g_output <= 0.0: output_gain = torch.nn.init.calculate_gain('linear')
                else: output_gain = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.orthogonal_(layer.weight, output_gain)

        elif self._w_init == "sparse":
            assert self._p_sparse < 1. and self._p_sparse >= 0.0

            # Construct initialization function:
            def init_hidden(layer):
                p_non_zero = self._p_sparse
                hidden_std = self._g_hidden

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, hidden_std)

            def init_output(layer):
                p_non_zero = self._p_sparse
                output_std = self._g_output

                torch.nn.init.constant_(layer.bias, self._b0)
                torch.nn.init.sparse_(layer.weight, p_non_zero, output_std)

        else:
            raise ValueError("Weight Initialization Type must be in ['xavier_normal', 'orthogonal', 'sparse'] but is {0}".format(self._w_init))

        # Compute In- / Output Sizes:
        self.n_dof = n_dof
        self.m = int((n_dof ** 2 + n_dof) / 2)

        # Compute non-zero elements of L:
        l_output_size = int((self.n_dof ** 2 + self.n_dof) / 2)
        l_lower_size = l_output_size - self.n_dof

        # Calculate the indices of the diagonal elements of L:
        idx_diag = np.arange(self.n_dof) + 1
        idx_diag = idx_diag * (idx_diag + 1) / 2 - 1

        # Calculate the indices of the off-diagonal elements of L:
        idx_tril = np.extract([x not in idx_diag for x in np.arange(l_output_size)], np.arange(l_output_size))

        # Indexing for concatenation of l_o  and l_d
        cat_idx = np.hstack((idx_diag, idx_tril))
        order = np.argsort(cat_idx)
        self._idx = np.arange(cat_idx.size)[order]

        # create it once and only apply repeat, this may decrease memory allocation
        self._eye = torch.eye(self.n_dof).view(1, self.n_dof, self.n_dof)
        self.low_tri = LowTri(self.n_dof)

        # Create Network:
        self.layers = nn.ModuleList()
        non_linearity = kwargs.get("activation", "ReLu")

        # Create Input Layer:
        self.layers.append(LagrangianLayer(self.n_dof, self.n_width, activation=non_linearity))
        init_hidden(self.layers[-1])

        # Create Hidden Layer:
        for _ in range(1, self.n_hidden):
            self.layers.append(LagrangianLayer(self.n_width, self.n_width, activation=non_linearity))
            init_hidden(self.layers[-1])

        self.res_layers = nn.ModuleList()
        last_layer_d = self.n_width
        # Create Residual Layer:
        for i, layer in enumerate(self.residual_layers):
            self.res_layers.append(ResLayer(last_layer_d, layer, activation=True)) if i != len(
                self.residual_layers) - 1 else self.res_layers.append(ResLayer(last_layer_d, layer, activation=False))
            init_hidden(self.res_layers[-1])
            last_layer_d = layer

        # Create output Layer:
        self.net_g = LagrangianLayer(self.n_width, 1, activation="Linear")
        init_output(self.net_g)

        self.net_lo = LagrangianLayer(self.n_width, l_lower_size, activation="Linear")
        init_hidden(self.net_lo)

        # The diagonal must be non-negative. Therefore, the non-linearity is set to ReLu.
        self.net_ld = LagrangianLayer(self.n_width, self.n_dof, activation="ReLu")
        init_hidden(self.net_ld)
        torch.nn.init.constant_(self.net_ld.bias, self._b0_diag)

    def forward(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        dEdt = out[6] + out[7]

        return tau_pred, dEdt

    def _dyn_model(self, q, qd, qdd):
        qd_3d = qd.view(-1, self.n_dof, 1)
        qd_4d = qd.view(-1, 1, self.n_dof, 1)

        # Create initial derivative of dq/dq.
        der = self._eye.repeat(q.shape[0], 1, 1).type_as(q)

        # Compute shared network between l & g:
        y, der = self.layers[0](q, der)

        hidden_outputs = []
        for i in range(1, len(self.layers)):
            y, der = self.layers[i](y, der)
            hidden_outputs.append(y)

        res_out = hidden_outputs[-2] if len(hidden_outputs) > 1 else hidden_outputs[-1]
        for i in range(0, len(self.res_layers)):
            res_out = self.res_layers[i](res_out)

        # Compute the network heads including the corresponding derivative:
        l_lower, der_l_lower = self.net_lo(y, der)
        l_diag, der_l_diag = self.net_ld(y, der)

        # Compute the potential energy and the gravitational force:
        V, der_V = self.net_g(y, der)
        V = V.squeeze()
        g = der_V.squeeze()

        # Assemble l and der_l
        l_diag = l_diag
        l = torch.cat((l_diag, l_lower), 1)[:, self._idx]
        der_l = torch.cat((der_l_diag, der_l_lower), 1)[:, self._idx, :]

        # Compute H:
        L = self.low_tri(l)
        LT = L.transpose(dim0=1, dim1=2)
        H = torch.matmul(L, LT) + self._epsilon * torch.eye(self.n_dof).type_as(L)

        # Calculate dH/dt
        Ldt = self.low_tri(torch.matmul(der_l, qd_3d).view(-1, self.m))
        Hdt = torch.matmul(L, Ldt.transpose(dim0=1, dim1=2)) + torch.matmul(Ldt, LT)

        # Calculate dH/dq:
        Ldq = self.low_tri(der_l.transpose(2, 1).reshape(-1, self.m)).reshape(-1, self.n_dof, self.n_dof, self.n_dof)
        Hdq = torch.matmul(Ldq, LT.view(-1, 1, self.n_dof, self.n_dof)) + torch.matmul(L.view(-1, 1, self.n_dof, self.n_dof), Ldq.transpose(2, 3))

        # Compute the Coriolis & Centrifugal forces:
        Hdt_qd = torch.matmul(Hdt, qd_3d).view(-1, self.n_dof)
        quad_dq = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), torch.matmul(Hdq, qd_4d)).view(-1, self.n_dof)
        c = Hdt_qd - 1. / 2. * quad_dq

        # Compute the Torque using the inverse model:
        H_qdd = torch.matmul(H, qdd.view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        tau_pred = H_qdd + c + g + res_out

        # Compute kinetic energy T
        H_qd = torch.matmul(H, qd_3d).view(-1, self.n_dof)
        T = 1. / 2. * torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qd.view(-1, 1, self.n_dof, 1)).view(-1)

        # Compute dT/dt:
        qd_H_qdd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), H_qdd.view(-1, 1, self.n_dof, 1)).view(-1)
        qd_Hdt_qd = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), Hdt_qd.view(-1, 1, self.n_dof, 1)).view(-1)
        dTdt = qd_H_qdd + 0.5 * qd_Hdt_qd

        # Compute dV/dt
        dVdt = torch.matmul(qd_4d.transpose(dim0=2, dim1=3), g.view(-1, 1, self.n_dof, 1)).view(-1)
        return tau_pred, H, c, g, T, V, dTdt, dVdt

    def inv_dyn(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        tau_pred = out[0]
        return tau_pred

    def for_dyn(self, q, qd, tau):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        H, c, g = out[1], out[2], out[3]

        # Compute Acceleration, e.g., forward model:
        invH = torch.inverse(H)
        qdd_pred = torch.matmul(invH, (tau - c - g).view(-1, self.n_dof, 1)).view(-1, self.n_dof)
        return qdd_pred

    def energy(self, q, qd):
        out = self._dyn_model(q, qd, torch.zeros_like(q))
        E = out[4] + out[5]
        return E

    def energy_dot(self, q, qd, qdd):
        out = self._dyn_model(q, qd, qdd)
        dEdt = out[6] + out[7]
        return dEdt

    def cuda(self, device=None):

        # Move the Network to the GPU:
        super(DeepResidualLagrangianNetwork, self).cuda(device=device)

        # Move the eye matrix to the GPU:
        self._eye = self._eye.cuda()
        self.device = self._eye.device
        return self

    def cpu(self):

        # Move the Network to the CPU:
        super(DeepResidualLagrangianNetwork, self).cpu()

        # Move the eye matrix to the CPU:
        self._eye = self._eye.cpu()
        self.device = self._eye.device
        return self

