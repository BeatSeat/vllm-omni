# Implementation adapted from https://github.com/EdwardDixon/snake under the MIT license.
#   LICENSE is in incl_licenses directory.

import torch
from torch import nn, pow, sin
from torch.nn import Parameter
from vllm.logger import init_logger

logger = init_logger(__name__)


class Snake(nn.Module):
    '''
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snake(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha: trainable parameter
            alpha is initialized to 1 by default, higher values = higher-frequency.
            alpha will be trained along with the rest of your model.
        '''
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        Snake ∶= x + 1/a * sin^2 (xa)
        '''
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
        x = x + (1.0 / (alpha + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x


class SnakeBeta(nn.Module):
    _triton_kernel = None
    _TRITON_MAX_BLOCK_T = 4096

    '''
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - Modified from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    '''
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False):
        '''
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        '''
        super().__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
            self.beta = Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = Parameter(torch.ones(in_features) * alpha)
            self.beta = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001
        self.register_buffer("_alpha_scale", None, persistent=False)
        self.register_buffer("_inv_beta_scale", None, persistent=False)

    @staticmethod
    def _init_triton():
        if SnakeBeta._triton_kernel is not None:
            return SnakeBeta._triton_kernel is not False
        try:
            import triton
            import triton.language as tl
        except ImportError:
            SnakeBeta._triton_kernel = False
            return False

        @triton.jit
        def _kernel(  # noqa: N803
            x_ptr,
            alpha_ptr,
            inv_beta_ptr,
            out_ptr,
            stride_b,
            stride_c,
            t_len,
            block_t: tl.constexpr,
        ):
            bid = tl.program_id(0)
            cid = tl.program_id(1)
            t_off = tl.program_id(2) * block_t + tl.arange(0, block_t)
            mask = t_off < t_len

            x = tl.load(x_ptr + bid * stride_b + cid * stride_c + t_off, mask=mask, other=0.0)
            alpha = tl.load(alpha_ptr + cid)
            inv_beta = tl.load(inv_beta_ptr + cid)
            sin_val = tl.sin(x * alpha)
            out = x + inv_beta * sin_val * sin_val

            tl.store(out_ptr + bid * stride_b + cid * stride_c + t_off, out, mask=mask)

        SnakeBeta._triton_kernel = _kernel
        return True

    def precompute_exp_cache(self):
        with torch.no_grad():
            if self.alpha_logscale:
                alpha = torch.exp(self.alpha)
                beta = torch.exp(self.beta)
            else:
                alpha = self.alpha
                beta = self.beta
            self._alpha_scale = alpha.contiguous()
            self._inv_beta_scale = (1.0 / (beta + self.no_div_by_zero)).contiguous()

    def forward(self, x):
        '''
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        '''
        if x.is_cuda and not torch.is_grad_enabled() and self._init_triton():
            try:
                return self._triton_forward(x)
            except Exception:
                logger.warning("IndexTTS2 BigVGAN Triton SnakeBeta failed, falling back to eager", exc_info=True)
                SnakeBeta._triton_kernel = False
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(sin(x * alpha), 2)

        return x

    def _triton_forward(self, x):
        import triton

        if self._alpha_scale is None or self._inv_beta_scale is None:
            self.precompute_exp_cache()

        x = x.contiguous()
        B, C, T = x.shape
        out = torch.empty_like(x)
        block_t = min(triton.next_power_of_2(T), self._TRITON_MAX_BLOCK_T)
        self._triton_kernel[(B, C, triton.cdiv(T, block_t))](
            x,
            self._alpha_scale,
            self._inv_beta_scale,
            out,
            x.stride(0),
            x.stride(1),
            t_len=T,
            block_t=block_t,
        )
        return out
