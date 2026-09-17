# This file is adapted from ClearerVoice-Studio
# (https://github.com/modelscope/ClearerVoice-Studio),
# licensed under the Apache License 2.0.
# Original work Copyright (c) Alibaba Group.
# Modifications Copyright (c) 2026 <your name / institution>.


import torch
import torch.nn as nn
import torch.nn.functional as F


class Extractor(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.N = args.extractor.N
        self.L = args.extractor.L
        self.B = args.extractor.B
        self.K = args.extractor.K
        self.R = args.extractor.R

        self.mixture_encoder = MixtureEncoder(self.L, self.N)
        # Keep the historical 48-dim branch by default.  The isolated
        # single-branch capacity control sets ``eeg_hidden=96``; every other
        # architectural component is unchanged.
        self.eeg_encoder = EEGEncoder(hidden=int(getattr(args, 'eeg_hidden', 48)))
        self.decoder = Decoder(self.N, self.L)

        self.layer_norm = nn.GroupNorm(1, self.N, eps=1e-8)
        self.bottleneck_conv1x1 = nn.Conv1d(self.N, self.B, 1, bias=False)
        self.dual_rnn = nn.ModuleList([])

        for _ in range(self.R):
            self.dual_rnn.append(
                Dual_RNN_Block(
                    self.B, args.extractor.H,
                    rnn_type='LSTM',
                    dropout=0,
                    bidirectional=True,
                )
            )

        self.prelu = nn.PReLU()
        self.mask_conv1x1 = nn.Conv1d(self.B, self.N, 1, bias=False)
        self.eeg_hidden = self.eeg_encoder.hidden
        self.front_fusion = nn.Conv1d(self.B + self.eeg_hidden, self.B, 1, bias=False)
        self.block_fusion = nn.ModuleList([
            nn.Conv1d(self.B + self.eeg_hidden, self.B, 1, bias=False)
            for _ in range(self.R)
        ])

    def forward(self, mixture, eeg):
        mixture_w = self.mixture_encoder(mixture)
        est_mask = self._estimate_mask(mixture_w, eeg)
        est_source = self.decoder(mixture_w, est_mask)

        # T changed after conv1d in encoder, fix it here
        T_origin = mixture.size(-1)
        T_conv = est_source.size(-1)

        if T_conv < T_origin:
            est_source = F.pad(est_source, (0, T_origin - T_conv))
        elif T_conv > T_origin:
            est_source = est_source[..., :T_origin]

        y_hat = est_source.squeeze(1)
        return y_hat

    def _estimate_mask(self, mixture_w, eeg):
        '''
        Args:
            mixture_w: [M, N, D], M is batch size
            eeg: [M, T_eeg, C_eeg]
        returns:
            est_mask: [M, 1, N, D]
        '''
        M, _, D = mixture_w.size()

        y = self.layer_norm(mixture_w)
        y = self.bottleneck_conv1x1(y)

        eeg_feat = self.eeg_encoder(eeg).transpose(1,2)
        eeg_feat = F.interpolate(eeg_feat, size=D, mode='linear', align_corners=False)

        y  = self.front_fusion(torch.cat((y, eeg_feat), dim=1))
        y, gap = self._segmentation(y, self.K)
        eeg_seg, _ = self._segmentation(eeg_feat, self.K)
        eeg_flat = eeg_seg.flatten(2)

        for rnn, fusion in zip(self.dual_rnn, self.block_fusion):
            y = rnn(y)
            K, S = y.size(2), y.size(3)
            y_flat = y.flatten(2)
            y = fusion(torch.cat((y_flat, eeg_flat), dim=1)).reshape(M, self.B, K, S)

        y = self._over_add(y, gap)
        y = self.prelu(y)
        y = self.mask_conv1x1(y)
        est_mask = F.relu(y).unsqueeze(1)
        return est_mask

    def _padding(self, input, K):
        '''
           padding the audio times
           K: chunks of length
           P: hop size
           input: [B, N, L]
        '''
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K

        if gap > 0:
            pad = input.new_zeros(B, N, gap)
            input = torch.cat([input, pad], dim=2)

        _pad = input.new_zeros(B, N, P)
        input = torch.cat([_pad, input, _pad], dim=2)
        return input, gap

    def _segmentation(self, input, K):
        '''
           the segmentation stage splits
           K: chunks of length
           P: hop size
           input: [B, N, L]
           output: [B, N, K, S]
        '''
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = torch.cat([input1, input2], dim=3).view(
            B, N, -1, K
        ).transpose(2, 3)

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        '''
           Merge sequence
           input: [B, N, K, S]
           gap: padding length
           output: [B, N, L]
        '''
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input


class MixtureEncoder(nn.Module):
    def __init__(self, L, N):
        super().__init__()
        self.conv1d_U = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)

    def forward(self, mixture):
        '''
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        '''
        mixture = torch.unsqueeze(mixture, 1)  # [M, 1, T]
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class Decoder(nn.Module):
    def __init__(self, N, L):
        super().__init__()
        self.N, self.L = N, L
        self.basis_signals = nn.Linear(N, L, bias=False)

    @staticmethod
    def overlap_and_add(signal, frame_step):
        outer_dims = signal.size()[:-2]
        num_frames, frame_length = signal.size()[-2:]

        output_length = frame_step * (num_frames - 1) + frame_length
        framed = signal.reshape(-1, num_frames, frame_length).transpose(1, 2)

        reconstructed = F.fold(
            framed,
            output_size=(1, output_length),
            kernel_size=(1, frame_length),
            stride=(1, frame_step),
        )  # (batch, 1, 1, output_length)

        reconstructed = reconstructed.view(*outer_dims, output_length)
        return reconstructed

    def forward(self, mixture_w, est_mask):
        '''
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        '''
        M, C, N, K = est_mask.shape
        mixture_w = mixture_w.unsqueeze(1) # [M, 1, N, K]
        masked = mixture_w * est_mask # [M, C, N, K]
        masked = masked.reshape(M * C, N, K).transpose(1, 2) # [M*C, K, N]
        masked = self.basis_signals(masked) # [M*C, K, L]

        est = self.overlap_and_add(masked, self.L // 2) # [M*C, T]
        est = est.view(M, C, -1)
        return est


class Dual_RNN_Block(nn.Module):
    '''
       Implementation of the intra-RNN and the inter-RNN
       input:
            out_channels: The number of features in the hidden state h
            rnn_type: RNN, LSTM, GRU
            dropout: If non-zero, introduces a Dropout layer on the outputs
                     of each LSTM layer except the last layer,
                     with dropout probability equal to dropout. Default: 0
            bidirectional: If True, becomes a bidirectional LSTM. Default: False
    '''

    def __init__(
            self,
            out_channels,
            hidden_channels,
            rnn_type='LSTM',
            dropout=0,
            bidirectional=False,
        ):
        super().__init__()

        # RNN model
        self.intra_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.inter_rnn = getattr(nn, rnn_type)(
            out_channels, hidden_channels, 1,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )

        # Norm
        self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)

        # Linear
        self.intra_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            out_channels,
        )
        self.inter_linear = nn.Linear(
            hidden_channels * 2 if bidirectional else hidden_channels,
            out_channels,
        )

    def forward(self, x):
        '''
           x: [B, N, K, S]
           out: [B, N, K, S]
        '''
        B, N, K, S = x.shape

        # intra RNN
        # [BS, K, N]
        intra_rnn = x.permute(0, 3, 2, 1).contiguous().view(B*S, K, N)
        # [BS, K, H]
        intra_rnn, _ = self.intra_rnn(intra_rnn)
        # [BS, K, N]
        intra_rnn = self.intra_linear(intra_rnn.contiguous().view(B*S*K, -1)).view(B*S, K, -1)
        # [B, S, K, N]
        intra_rnn = intra_rnn.view(B, S, K, N)
        # [B, N, K, S]
        intra_rnn = intra_rnn.permute(0, 3, 2, 1).contiguous()
        intra_rnn = self.intra_norm(intra_rnn)

        # [B, N, K, S]
        intra_rnn = intra_rnn + x

        # inter RNN
        # [BK, S, N]
        inter_rnn = intra_rnn.permute(0, 2, 3, 1).contiguous().view(B*K, S, N)
        # [BK, S, H]
        inter_rnn, _ = self.inter_rnn(inter_rnn)
        # [BK, S, N]
        inter_rnn = self.inter_linear(inter_rnn.contiguous().view(B*S*K, -1)).view(B*K, S, -1)
        # [B, K, S, N]
        inter_rnn = inter_rnn.view(B, K, S, N)
        # [B, N, K, S]
        inter_rnn = inter_rnn.permute(0, 3, 1, 2).contiguous()
        inter_rnn = self.inter_norm(inter_rnn)
        # [B, N, K, S]
        out = inter_rnn + intra_rnn

        return out


class EEGEncoder(nn.Module):
    def __init__(
        self,
        n_ch: int = 64,
        hidden: int = 48,
        kernel: int = 15,
        dilations=[1, 2, 4, 8],
        p_drop: float = 0.15,
    ):
        super().__init__()
        self.hidden = hidden
        self.kernel = kernel
        self.dilations = dilations

        self.spatial = nn.Conv1d(n_ch, hidden, kernel_size=1, bias=False)

        self.dw = nn.ModuleList([
            nn.Conv1d(hidden, hidden, kernel_size=kernel,
                      dilation=d, groups=hidden, padding=0, bias=False)
            for d in self.dilations
        ])
        self.gn  = nn.ModuleList([nn.GroupNorm(1, hidden) for _ in self.dilations])
        self.pw  = nn.ModuleList([nn.Conv1d(hidden, hidden, kernel_size=1, bias=True)
                                   for _ in self.dilations])
        self.drop = nn.ModuleList([nn.Dropout(p_drop) for _ in self.dilations])
        self.act = nn.SiLU()

    def _right_pad(self, x: torch.Tensor, right: int) -> torch.Tensor:
        if right <= 0:
            return x

        mode = 'reflect' if right < x.size(-1) else 'replicate'
        return F.pad(x, (0, right), mode=mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, T, C_eeg)
        return: (B, T, hidden)
        """
        x = x.transpose(1, 2) # (B, T, C_eeg) -> (B, C_eeg, T)
        y = self.spatial(x)

        for i, d in enumerate(self.dilations):
            right = d * (self.kernel - 1)
            z = self._right_pad(y, right)
            z = self.dw[i](z)
            z = self.gn[i](z)
            z = self.act(z)
            z = self.pw[i](z)
            z = self.drop[i](z)
            y = y + z

        z = y.transpose(1, 2) # (B, hidden, T) -> (B, T, hidden)
        z = F.normalize(z, p=2, dim=-1)
        return z


class AudioEncoder(nn.Module):
    def __init__(
            self,
            in_channels=1,
            base_channels=32,
            d_model=48,
        ):
        super().__init__()

        def conv_block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.GroupNorm(num_groups=cout // 4, num_channels=cout),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(2, 1))  # halve F
            )

        self.conv1 = conv_block(in_channels, base_channels)        # -> (B, 32, F/2, T)
        self.conv2 = conv_block(base_channels, base_channels * 2)  # -> (B, 64, F/4, T)
        self.conv3 = conv_block(base_channels * 2, base_channels * 4)  # -> (B, 128, F/8, T)

        self.c_out = base_channels * 4

        def temp_block(ch, dilation):
            return nn.Sequential(
                nn.Conv1d(ch, ch, kernel_size=5, padding=2*dilation,
                          dilation=dilation, bias=False),
                nn.BatchNorm1d(ch),
                nn.GELU(),
                nn.Conv1d(ch, ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(ch)
            )

        self.tblock1 = temp_block(self.c_out, dilation=1)
        self.tblock2 = temp_block(self.c_out, dilation=2)

        self.proj = nn.Sequential(
            nn.Linear(self.c_out, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model)
        )

    def forward(self, x):
        # x : (B, 1, F_mel, T_mel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # Collapse frequency dimension -> mean over F
        x = x.mean(dim=2)   # -> (B, C, T)

        # Temporal conv blocks
        for block in [self.tblock1, self.tblock2]:
            residual = x
            x = block(x)
            x = F.gelu(x + residual)

        x = x.transpose(1, 2)       # -> (B, T, C)
        x = self.proj(x)
        x = F.normalize(x, p=2, dim=-1)

        return x
