import os

import torch

from ctc_forced_aligner.align import TORCH_DTYPES
from ctc_forced_aligner.alignment_utils import load_alignment_model


def main() -> None:
    device = os.getenv("ALIGN_DEVICE", "cpu")
    model_name = os.getenv("ALIGN_MODEL", "MahmoudAshraf/mms-300m-1130-forced-aligner")
    attn_implementation = os.getenv("ALIGN_ATTN_IMPLEMENTATION", "") or None
    compute_dtype = os.getenv("ALIGN_COMPUTE_DTYPE", "float32")
    dtype = TORCH_DTYPES.get(compute_dtype, torch.float32)

    load_alignment_model(
        device=device,
        model_path=model_name,
        attn_implementation=attn_implementation,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
