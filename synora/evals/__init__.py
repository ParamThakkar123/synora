from typing import Any


def __getattr__(name: str) -> Any:
    if name == "FID":
        from synora.evals.fid import FID

        return FID
    if name == "FVD":
        from synora.evals.fvd import FVD

        return FVD
    if name == "LPIPS":
        from synora.evals.lpips import LPIPS

        return LPIPS
    if name == "PSNR":
        from synora.evals.psnr import PSNR

        return PSNR
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["FID", "FVD", "LPIPS", "PSNR"]
