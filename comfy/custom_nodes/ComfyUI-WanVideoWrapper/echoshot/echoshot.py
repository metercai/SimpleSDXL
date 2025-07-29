import torch
from comfy.model_management import get_autocast_device, get_torch_device

@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply_z(x, grid_sizes, freqs, inner_t, shift=6):
    n, c = x.size(2), x.size(3) // 2

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        start_ind = [sum(inner_t[i][:_]) for _ in range(len(inner_t[i]))]
        end_ind = [sum(inner_t[i][:_+1]) for _ in range(len(inner_t[i]))]

        freq_select = []
        for shot_ind, (s, e) in enumerate(zip(start_ind, end_ind)):
            freq_select += [shot_ind * shift] * (e - s)
        shot_freqs = freqs[freq_select]

        freqs_i = shot_freqs.view(f, 1, 1, -1).expand(f, h, w, -1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


@torch.autocast(device_type=get_autocast_device(get_torch_device()), enabled=False)
@torch.compiler.disable()
def rope_apply_c(x, freqs, inner_c, shift=6):

    b, s, n, c = x.size(0), x.size(1), x.size(2), x.size(3) // 2

    # loop over samples
    output = []
    for i in range(b):

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i].to(torch.float64).reshape(s, n, -1, 2)
        )

        freq_select = []
        for shot_ind, c_len in enumerate(inner_c[i]):
            freq_select += [shot_ind * shift] * c_len
        freq_select += [shot_ind+10] * (s-len(freq_select)) # extra suppression for the empty token
        shot_freqs = freqs[freq_select]

        freqs_i = shot_freqs.view(s, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()