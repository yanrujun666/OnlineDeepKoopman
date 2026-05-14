import torch
import torch.nn as nn


def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega


class Koopmanv1(nn.Module):
    """
    Deep Neural Networks With Koopman Operators for Modeling and Control of Autonomous Vehicles, T-IV 2023
    """

    def __init__(self, state_dim, hidden_dim, control_dim):
        super(Koopmanv1, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(1 / control_dim * torch.zeros(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class Koopmanv1MoE(nn.Module):
    """
    Mixture of Koopmans
    """

    def __init__(self, state_dim, hidden_dim, control_dim, condition_num=1):
        super(Koopmanv1MoE, self).__init__()
        self.state_dim = state_dim
        self.As = nn.Parameter(torch.zeros(condition_num, hidden_dim, hidden_dim), requires_grad=True) # C hidden_dim hidden_dim
        self.As.data = torch.eye(hidden_dim, hidden_dim).unsqueeze(0).repeat(condition_num, 1, 1) # init
        self.Bs = nn.Parameter(1 / control_dim * torch.zeros(condition_num, hidden_dim, control_dim), requires_grad=True) # C control_dim hidden_dim

    def forward(self, x, u, c, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
            c: B C
        Output:
            results: B S hidden_dim
        """
        A = torch.einsum('bc,cnm->bnm', c, self.As)
        B = torch.einsum('bc,cnm->bnm', c, self.Bs)
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            # x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            x_tmp = torch.einsum('bnm,bn->bm', A, x_tmp) + torch.einsum('bmc,bc->bm', B, u_tmp) # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class Koopmanv1MoEv2(nn.Module):
    """
    Mixture of Koopman v2
    """

    def __init__(self, state_dim, hidden_dim, control_dim, condition_num=1):
        super(Koopmanv1MoEv2, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(hidden_dim, control_dim), requires_grad=True)
        self.As = nn.Parameter(torch.zeros(condition_num, hidden_dim, hidden_dim), requires_grad=True) # C hidden_dim hidden_dim
        self.Bs = nn.Parameter(torch.zeros(condition_num, hidden_dim, control_dim), requires_grad=True) # C control_dim hidden_dim

    def forward(self, x, u, c, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
            c: B C
        Output:
            results: B S hidden_dim
        """
        A = torch.einsum('bc,cnm->bnm', c, self.As) + self.A
        B = torch.einsum('bc,cnm->bnm', c, self.Bs) + self.B
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            # x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            x_tmp = torch.einsum('bnm,bn->bm', A, x_tmp) + torch.einsum('bmc,bc->bm', B, u_tmp) # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class Koopmanv1MoEv3(nn.Module):
    """
    Mixture of Koopman v3
    """

    def __init__(self, state_dim, hidden_dim, control_dim, condition_num=1):
        super(Koopmanv1MoEv3, self).__init__()
        self.state_dim = state_dim
        self.As = nn.Parameter(torch.zeros(condition_num, hidden_dim, hidden_dim), requires_grad=True) # C hidden_dim hidden_dim
        self.As.data[0] = torch.eye(hidden_dim, hidden_dim)
        self.As.data[1:] = torch.zeros(hidden_dim, hidden_dim).unsqueeze(0).repeat(condition_num-1, 1, 1) # init
        self.Bs = nn.Parameter(torch.zeros(condition_num, hidden_dim, control_dim), requires_grad=True) # C control_dim hidden_dim
    
    def forward(self, x, u, c, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
            c: B C
        Output:
            results: B S hidden_dim
        """
        cnew = c.clone()
        mask = (c[:,0] == 0)
        cnew[mask, 0] += 1
        assert (cnew[:, 0] > 0).all()

        A = torch.einsum('bc,cnm->bnm', cnew, self.As)
        B = torch.einsum('bc,cnm->bnm', cnew, self.Bs)
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            # x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            x_tmp = torch.einsum('bnm,bn->bm', A, x_tmp) + torch.einsum('bmc,bc->bm', B, u_tmp) # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class Koopmanv1MoEv4(nn.Module):
    """
    Mixture of Koopman v4
    """

    def __init__(self, state_dim, hidden_dim, control_dim, condition_num=1):
        super(Koopmanv1MoEv4, self).__init__()
        self.state_dim = state_dim
        self.As = nn.Parameter(torch.zeros(condition_num, hidden_dim, hidden_dim), requires_grad=True) # C hidden_dim hidden_dim
        self.As.data = torch.eye(hidden_dim, hidden_dim).unsqueeze(0).repeat(condition_num, 1, 1) # init
        self.Bs = nn.Parameter(1 / control_dim * torch.zeros(condition_num, hidden_dim, control_dim), requires_grad=True) # C control_dim hidden_dim

    def forward(self, x, u, c, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
            c: B C
        Output:
            results: B S hidden_dim
        """
        c = torch.nn.functional.softmax(c, dim=1)
        A = torch.einsum('bc,cnm->bnm', c, self.As)
        B = torch.einsum('bc,cnm->bnm', c, self.Bs)
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            # x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            x_tmp = torch.einsum('bnm,bn->bm', A, x_tmp) + torch.einsum('bmc,bc->bm', B, u_tmp) # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class Koopmanv1MoEv5(nn.Module):
    """
    Mixture of Koopman v5
    """

    def __init__(self, state_dim, hidden_dim, control_dim, condition_num=1):
        super(Koopmanv1MoEv5, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(hidden_dim, control_dim), requires_grad=True)
        self.As = nn.Parameter(torch.zeros(condition_num, hidden_dim, hidden_dim), requires_grad=True) # C hidden_dim hidden_dim
        self.Bs = nn.Parameter(torch.zeros(condition_num, hidden_dim, control_dim), requires_grad=True) # C control_dim hidden_dim

        nn.init.normal_(self.As, mean=0.0, std=0.001/hidden_dim)
        nn.init.normal_(self.Bs, mean=0.0, std=0.001/control_dim)
        # print(self.As, self.Bs)

    def forward(self, x, u, c, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
            c: B C
        Output:
            results: B S hidden_dim
        """
        A = torch.einsum('bc,cnm->bnm', c, self.As) + self.A
        B = torch.einsum('bc,cnm->bnm', c, self.Bs) + self.B
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            # x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            x_tmp = torch.einsum('bnm,bn->bm', A, x_tmp) + torch.einsum('bmc,bc->bm', B, u_tmp) # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class Koopmanv1MoERI(nn.Module):
    """
    Deep Neural Networks With Koopman Operators for Modeling and Control of Autonomous Vehicles, T-IV 2023
    """

    def __init__(self, state_dim, hidden_dim, control_dim, condition_num=1):
        super(Koopmanv1MoERI, self).__init__()
        self.state_dim = state_dim
        self.As = nn.Parameter(torch.zeros(condition_num, hidden_dim, hidden_dim), requires_grad=True) # C hidden_dim hidden_dim
        init_data = []
        for _ in range(condition_num):
            init_data.append(torch.eye(hidden_dim, hidden_dim) + 1 / hidden_dim * torch.randn(hidden_dim, hidden_dim) * 0.01)
        self.As.data = torch.stack(init_data, dim=0) # init
        self.Bs = nn.Parameter(1 / control_dim * torch.randn(condition_num, hidden_dim, control_dim) * 0.001, requires_grad=True) # C control_dim hidden_dim

        print(self.As.data.shape, self.Bs.data.shape)
        print(self.As, self.Bs)

    def forward(self, x, u, c, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
            c: B C
        Output:
            results: B S hidden_dim
        """
        A = torch.einsum('bc,cnm->bnm', c, self.As)
        B = torch.einsum('bc,cnm->bnm', c, self.Bs)
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            # x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            x_tmp = torch.einsum('bnm,bn->bm', A, x_tmp) + torch.einsum('bmc,bc->bm', B, u_tmp) # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
        

class Koopmanv1RI(nn.Module):
    """
    Deep Neural Networks With Koopman Operators for Modeling and Control of Autonomous Vehicles, T-IV 2023
    """

    def __init__(self, state_dim, hidden_dim, control_dim):
        super(Koopmanv1RI, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.eye(hidden_dim, hidden_dim) + 1e-3 * torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(1e-3 * torch.randn(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)


class Koopmanv2(nn.Module):
    """
    """

    def __init__(self, state_dim, hidden_dim, control_dim):
        super(Koopmanv2, self).__init__()
        self.state_dim = state_dim
        self.A = nn.Parameter(torch.zeros(hidden_dim, hidden_dim), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(control_dim, hidden_dim), requires_grad=True)
        self.dt = 0.01

    def forward(self, x, u, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = (x_tmp @ self.A + u_tmp @ self.B) * self.dt + x_tmp # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)


class ResKoopman(nn.Module):

    def __init__(self, state_dim, hidden_dim, control_dim):
        super(ResKoopman, self).__init__()
        self.state_dim = state_dim
        self.As = nn.Parameter(torch.eye(state_dim, state_dim), requires_grad=True)
        self.Bs = nn.Parameter(torch.zeros(control_dim, state_dim), requires_grad=True)

        self.Af = nn.Parameter(torch.zeros(hidden_dim, hidden_dim), requires_grad=True)
        self.Bf = nn.Parameter(torch.zeros(control_dim, hidden_dim), requires_grad=True)

        self.dt = 0.01

    def forward(self, x, u, iters=1):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp_rule = x_tmp[:, self.state_dim] @ self.As + u_tmp @ self.Bs # B state_dim
            x_tmp_res = x_tmp @ self.Af + u_tmp @ self.Bf # B hidden_dim
            x_tmp = x_tmp_res
            x_tmp[:, self.state_dim] += x_tmp_rule # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

def _orth(w):
    return torch.linalg.qr(w)[0]


class OrthogonalViaExponential(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.A = nn.Parameter(torch.randn(n, n))
    
    def forward(self):
        skew = self.A - self.A.t()  # 斜对称矩阵
        return torch.matrix_exp(skew)  # 指数映射


class StrictStableKoopman(nn.Module):

    def __init__(self, state_dim, hidden_dim, control_dim):
        super().__init__()

        self.U_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.V_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.S_raw = nn.Parameter(torch.randn(hidden_dim), requires_grad=True)
        self.p_max = 0.95

        self.B = nn.Parameter(1 / control_dim * torch.randn(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        U, V = _orth(self.U_raw), _orth(self.V_raw)
        zeta = torch.sigmoid(self.S_raw) * self.p_max
        # zeta = (torch.tanh(self.S_raw) + 1) / 2 * self.p_max
        A = U @ torch.diag(zeta) @ V.T

        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class StrictStableKoopmanv2(nn.Module):

    def __init__(self, state_dim, hidden_dim, control_dim):
        super().__init__()

        self.U_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.V_raw = nn.Parameter(torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.S_raw = nn.Parameter(torch.ones(hidden_dim), requires_grad=True)
        # self.A = nn.Parameter(1 / hidden_dim * torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.p_max = 0.95

        self.B = nn.Parameter(1 / control_dim * torch.zeros(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        U = torch.matrix_exp(self.U_raw - self.U_raw.t())
        V = torch.matrix_exp(self.V_raw - self.V_raw.t())
        zeta = (torch.tanh(self.S_raw) + 1) / 2 * self.p_max
        A = U @ torch.diag(zeta) @ V.T

        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class ResdiualKoopman(nn.Module):

    def __init__(self, state_dim, hidden_dim, control_dim):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.dt = 0.01

        self.Ab = nn.Parameter(torch.eye(state_dim, state_dim), requires_grad=True)
        self.Bb = nn.Parameter(torch.zeros(control_dim, state_dim), requires_grad=True)

        self.Ar = nn.Parameter(0.0001 * torch.randn(hidden_dim, hidden_dim), requires_grad=True)
        self.Br = nn.Parameter(0.0001 * torch.randn(control_dim, hidden_dim), requires_grad=True)

    def forward(self, x, u, iters):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        b, s, c = x.shape
        
        A = torch.zeros_like(self.Ar)
        A = A + self.Ar
        A[:self.state_dim, :self.state_dim] = A[:self.state_dim, :self.state_dim] + self.Ab
        B = torch.zeros_like(self.Br)
        B = B + self.Br
        B[:, :self.state_dim] = B[:, :self.state_dim] + self.Bb

        results = []
        results_b = []
        x_tmp = x[:, 0, :]
        x_b_tmp = x[:, 0, :self.state_dim]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ A + u_tmp @ B # B hidden_dim
            x_b_tmp = x_b_tmp @ self.Ab + u_tmp @ self.Bb # B state_dim
            results.append(x_tmp)
            results_b.append(x_b_tmp)
        return torch.stack(results, dim=1), torch.stack(results_b, dim=1)
    

class ResdiualKoopmanv2(nn.Module):

    def __init__(self, state_dim, hidden_dim, control_dim):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.dt = 0.01

        self.Ab = nn.Parameter(torch.eye(state_dim, state_dim), requires_grad=True)
        self.B = nn.Parameter(torch.zeros(control_dim, hidden_dim), requires_grad=True)

        self.Ar = nn.Parameter(0.0001 * torch.randn(hidden_dim-state_dim, hidden_dim), requires_grad=True)

        A = torch.zeros(hidden_dim, hidden_dim)
        A[:state_dim, :state_dim] = self.Ab
        A[state_dim:, :] = self.Ar
        self.A = A.T

    def forward(self, x, u, iters):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        b, s, c = x.shape
        
        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ self.A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)
    

class EigenKoopman(nn.Module):

    def __init__(self, state_dim, hidden_dim, control_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_complex_pairs = hidden_dim // 2
        self.num_real = hidden_dim % 2

        if self.num_complex_pairs > 0:
            self.complex_real = nn.Parameter(torch.ones(self.num_complex_pairs))  # 实部
            self.complex_imag = nn.Parameter(torch.zeros(self.num_complex_pairs))  # 虚部
        if self.num_real > 0:
            self.real_eigvals = nn.Parameter(torch.ones(self.num_real))

        self.B = nn.Parameter(1 / control_dim * torch.zeros(control_dim, hidden_dim), requires_grad=True)
    
    def build_A(self):
        A = torch.zeros(self.hidden_dim, self.hidden_dim)
        idx = 0

        # 共轭复数对
        for j in range(self.num_complex_pairs):
            C = self.complex_real[j]
            P = self.complex_imag[j]
            block = torch.tensor([[C, P],
                                   [-P, C]], dtype=torch.float32)
            A[idx:idx+2, idx:idx+2] = block
            idx += 2

        # 实数特征值
        for k in range(self.num_real):
            A[idx, idx] = self.real_eigvals[k]
            idx += 1

        return A

    def forward(self, x, u, iters):
        """
        Input:
            x: B S hidden_dim
            u: B S control_dim
        Output:
            results: B S hidden_dim
        """
        A = self.build_A().to(x.device)

        results = []
        x_tmp = x[:, 0, :]
        for i in range(iters):
            u_tmp = u[:, i, :]
            x_tmp = x_tmp @ A + u_tmp @ self.B # B hidden_dim
            results.append(x_tmp)
        return torch.stack(results, dim=1)