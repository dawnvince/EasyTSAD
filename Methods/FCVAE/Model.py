import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

class EncoderLayer_selfattn(nn.Module):
    """Compose with two layers"""

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer_selfattn, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class EncoderLayer_selfattn2(nn.Module):
    """Compose with two layers"""
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer_selfattn, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input):
        enc_output, enc_slf_attn = self.slf_attn(enc_input[:,-1,:].unsqueeze(1), enc_input[:,:-1,:], enc_input[:,:-1,:])
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()
        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)
        output, attn = self.attention(q, k, v)
        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)
        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)
        return output, attn


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """A two-feed-forward-layer module"""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1)
        self.w_2 = nn.Conv1d(d_hid, d_in, 1)
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class CVAE(nn.Module):
    def __init__(
        self,
        hp,
        condition_emb_dim,
        latent_dim,
        in_channels,
        hidden_dims,
        step_max,
        window,
        batch_size,
        beta: int = 4,
        gamma: float = 1000.0,
        max_capacity: int = 25,
        Capacity_max_iter: int = 1e5,
        loss_type: str = "C",
    ):
        super(CVAE, self).__init__()
        self.eps = 1e-7
        self.hp = hp
        self.condition_emb_dim = 2*self.hp.condition_emb_dim
        self.latent_dim = latent_dim
        modules = []
        self.num_iter = 0
        self.step_max = step_max
        self.window = window
        self.batch_size = batch_size
        self.step_now = 0
        self.beta = beta
        self.gamma = gamma
        self.loss_type = loss_type
        self.C_max = torch.Tensor([max_capacity])
        self.C_stop_iter = Capacity_max_iter
        in_channels = window + self.condition_emb_dim
        if hidden_dims is None:
            hidden_dims = [100, 100]
        self.hidden_dims = hidden_dims
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.Tanh(),
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)
        self.bn = nn.BatchNorm1d(latent_dim)
        self.now_dim = int(self.window / (2 ** len(hidden_dims)))
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Sequential(
            nn.Linear(hidden_dims[-1], latent_dim),
            nn.Softplus(),
        )

        modules = []

        self.decoder_input = nn.Linear(
            latent_dim + self.condition_emb_dim, hidden_dims[-1]
        )

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.Tanh(),
                )
            )

        modules.append(
            nn.Sequential(
                nn.Linear(hidden_dims[-1], self.window),
                nn.Tanh(),
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.fc_mu_x = nn.Linear(self.window, self.window)
        self.fc_var_x = nn.Sequential(
            nn.Linear(self.window, self.window),
            nn.Softplus()
        )

        self.atten = nn.ModuleList(
            [
                EncoderLayer_selfattn(
                    self.hp.d_model,
                    self.hp.d_inner,
                    self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    self.hp.d_inner // self.hp.n_head,
                    dropout=0.1,
                )
                for _ in range(1)
            ]
        )
        self.in_linear = nn.Sequential(
            nn.Linear(2+self.hp.kernel_size, self.hp.d_model),
            nn.Tanh(),
        )
        self.condition_emb_dim = self.condition_emb_dim//2
        self.out_linear = nn.Sequential(
            nn.Linear(self.hp.d_model,self.condition_emb_dim),
            nn.Tanh(),
        )
        self.dropout =nn.Dropout(self.hp.dropout_rate)
        self.emb = nn.Sequential(
            nn.Linear(self.hp.window,self.condition_emb_dim),
            nn.Tanh(),
        )

    def encode(self, input):
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)
        mu = self.fc_mu(result)
        var = self.fc_var(result)
        return [mu, var]

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 1, self.hidden_dims[0])
        result = self.decoder(result)
        mu_x = self.fc_mu_x(result)
        var_x = self.fc_var_x(result)
        return mu_x, var_x

    def reparameterize(self, mu, var):
        std = torch.sqrt(1e-7 + var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input, mode, y_all):
        if mode == "train" or mode == "valid":
            condition = self.get_conditon(input)
            condition = self.dropout(condition)
            mu, var = self.encode(torch.cat((input, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            rec_x = self.reparameterize(mu_x, var_x)
            loss = self.loss_func(mu_x, var_x, input, mu, var, y_all, z)
            return [mu_x, var_x, rec_x, mu, var, loss]
        else:
            y_all = y_all.unsqueeze(1)
            x = input
            return self.MCMC2(x)

    def get_conditon(self, x):
        x_c =x
        f_global = torch.fft.rfft(x_c[:,:,:-1],dim=-1)
        f_global = torch.cat((f_global.real,f_global.imag),dim=-1)
        f_global = self.emb(f_global)
        x_c = x_c.view(x.shape[0], 1, 1, -1)
        x_c_l = x_c.clone()
        x_c_l[:,:,:,-1] = 0
        unfold = nn.Unfold(
            kernel_size=(1, self.hp.kernel_size),
            dilation=1,
            padding=0,
            stride=(1, self.hp.stride),
        )
        unfold_x = unfold(x_c_l)
        unfold_x = unfold_x.transpose(1, 2)
        freq = torch.fft.rfft(unfold_x, dim=-1)
        #np.save('./npy/smallwindowfrq_{}.npy'.format(self.hp.data_dir[7:]),(torch.abs(freq)).cpu().detach().numpy())
        freq = torch.cat((freq.real, freq.imag), dim=-1)

        enc_output = self.in_linear(freq)
        for enc_layer in self.atten:
            enc_output, enc_slf_attn = enc_layer(enc_output)
        # np.save('./npy/atten_{}.npy'.format(self.hp.data_dir[7:]),enc_slf_attn.cpu().detach().numpy())
        # np.save('./npy/origin_{}.npy'.format(self.hp.data_dir[7:]),x.cpu().detach().numpy())
        # np.save('./npy/smallwindow_{}.npy'.format(self.hp.data_dir[7:]),unfold_x.cpu().detach().numpy())
        enc_output = self.out_linear(enc_output)
        f_local = enc_output[:, -1, :].unsqueeze(1)

        output = torch.cat((f_global,f_local),-1)
        return output
    
    def MCMC2(self, x):
        condition = self.get_conditon(x)
        origin_x = x.clone()
        for i in range(10):
            mu, var = self.encode(torch.cat((x, condition), dim=2))
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            recon = -0.5 * (torch.log(var_x + self.eps) + (origin_x - mu_x) ** 2 / (var_x + self.eps))
            temp = (
                torch.from_numpy(np.percentile(recon.cpu(), self.hp.mcmc_rate, axis=-1))
                .unsqueeze(2)
                .repeat(1, 1, self.window)
            ).to("cuda")
            if(self.hp.mcmc_mode==0):
                l = (temp < recon).int()
                x = mu_x * (1 - l) + origin_x * l
            if(self.hp.mcmc_mode==1):
                l = (self.hp.mcmc_value < recon).int()
                x = origin_x * l+mu_x * (1 - l)
            if(self.hp.mcmc_mode==2):
                l = torch.ones_like(origin_x)
                l[:,:,-1]=0
                x = origin_x*l +(1-l)*mu_x
        prob_all = 0
        mu, var = self.encode(torch.cat((x, condition), dim=2))
        for i in range(128):
            z = self.reparameterize(mu, var)
            mu_x, var_x = self.decode(torch.cat((z, condition.squeeze(1)), dim=1))
            prob_all += -0.5 * (torch.log(var_x + self.eps) + (origin_x - mu_x) ** 2 / (var_x + self.eps))
        return x, prob_all / 128

    def loss_func(self, mu_x, var_x, input, mu, var, y_all, z, mode="nottrain"):
        if mode == "train":
            self.num_iter += 1
            self.num_iter = self.num_iter % 100
        kld_weight = 0.005
        mu_x = mu_x.squeeze(1)
        var_x = var_x.squeeze(1)
        input = input.squeeze(1)
        w = torch.zeros_like(mu_x)
        w[:,-1] = 5
        recon_loss = torch.mean(
            0.5
            * torch.mean(
                y_all * (torch.log(var_x + self.eps) + (input - mu_x) ** 2 / (var_x + self.eps)), dim=1
            ),
            dim=0,
        )
        m = (torch.sum(y_all, dim=1, keepdim=True) / self.window).repeat(
            1, self.latent_dim
        )
        kld_loss = torch.mean(
            0.5
            * torch.mean(m * (z**2) - torch.log(var + self.eps) - (z - mu) ** 2 / (var + self.eps), dim=1),
            dim=0,
        )
        if self.loss_type == "B":
            self.C_max = self.C_max.to(input.device)
            C = torch.clamp(
                self.C_max / self.C_stop_iter * self.num_iter, 0, self.C_max.data[0]
            )
            loss = recon_loss + self.gamma * kld_weight * (kld_loss - C).abs()
        elif self.loss_type == "C":
            loss = recon_loss + kld_loss
        elif self.loss_type == "D":
            loss = recon_loss + self.num_iter / 100 * kld_loss
        else:
            raise ValueError("Undefined loss type.")
        return loss
    
class FCVAEModel(nn.Module):
    def __init__(self, hparams) -> None:
        super(FCVAEModel, self).__init__()
        self.hp = hparams
        self.window = hparams.window
        self.latent_dim = hparams.latent_dim
        self.hidden_dims = None
        self.step_max = 0
        
        self.vae = CVAE(
                self.hp,
                self.hp.condition_emb_dim,
                self.latent_dim,
                1,
                self.hidden_dims,
                self.step_max,
                self.window,
                self.hp.batch_size,
            )
        
    def forward(self, x, mode, mask):
        x = x.view(-1, 1, self.window)
        return self.vae.forward(x, mode, mask)