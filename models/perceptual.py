import torch.nn.functional as F
from einops import einsum
import lpips


def gram_matrix(input):
    b, c, h, w = input.shape
    return einsum(input, input, "b c1 h w, b c2 h w -> b c1 c2") / (c * h * w)


class LPIPS(lpips.LPIPS):
    def forward(
        self,
        in0,
        in1,
        retPerLayer=False,
        normalize=False,
        ret_gram=False,
    ):
        if (
            normalize
        ):  # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0 - 1
            in1 = 2 * in1 - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            (self.scaling_layer(in0), self.scaling_layer(in1))
            if self.version == "0.1"
            else (in0, in1)
        )
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = lpips.normalize_tensor(
                outs0[kk]
            ), lpips.normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        if self.lpips:
            if self.spatial:
                res = [
                    lpips.upsample(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:])
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(self.lins[kk](diffs[kk]), keepdim=True)
                    for kk in range(self.L)
                ]
        else:
            if self.spatial:
                res = [
                    lpips.upsample(
                        diffs[kk].sum(dim=1, keepdim=True), out_HW=in0.shape[2:]
                    )
                    for kk in range(self.L)
                ]
            else:
                res = [
                    lpips.spatial_average(
                        diffs[kk].sum(dim=1, keepdim=True), keepdim=True
                    )
                    for kk in range(self.L)
                ]

        val = 0
        for l in range(self.L):
            val += res[l]

        gram = 0
        if ret_gram:
            for kk in range(self.L):
                gram_in0, gram_in1 = gram_matrix(outs0[kk]), gram_matrix(outs1[kk])
                gram += F.l1_loss(gram_in0, gram_in1)

        if retPerLayer:
            return ((val, res), gram)
        else:
            return (val, gram)
