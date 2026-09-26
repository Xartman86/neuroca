"""Best-effort визуализация многомерной машины (matplotlib/imageio опциональны).

Срезы: строки — биты регистра, столбцы — срезы по последней пространственной оси.
"""
from __future__ import annotations

import numpy as np

OK = False
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: N813
    OK = True
except Exception:
    plt = None

GIF_OK = False
try:
    import imageio.v2 as imageio  # type: ignore
    GIF_OK = True
except Exception:
    try:
        import imageio  # type: ignore
        GIF_OK = True
    except Exception:
        imageio = None


def save_state_grid(state, path, max_z=4, title=""):
    """PNG: сетка срезов — строки = биты регистра, столбцы = срезы по последней оси."""
    if not OK:
        return False
    s = np.asarray(state)
    N = s.ndim - 1
    W = s.shape[-1]
    Lz = s.shape[N - 1]
    zs = np.linspace(0, Lz - 1, min(max_z, Lz)).astype(int)

    fig, axes = plt.subplots(W, len(zs), squeeze=False,
                             figsize=(2.2 * len(zs), 2.0 * W))
    for bi in range(W):
        for cj, z in enumerate(zs):
            idx = (slice(None),) * (N - 1) + (int(z), bi)
            ax = axes[bi, cj]
            ax.imshow(s[idx], cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if bi == 0:
                ax.set_title(f"z={int(z)}", fontsize=8)
            if cj == 0:
                ax.set_ylabel(f"b={bi}", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(str(path), dpi=110)
    plt.close(fig)
    return True


def save_gif(states, path, fps=6):
    """GIF: средний срез по последней пространственной оси, биты — рядом по горизонтали."""
    if not GIF_OK:
        return False
    frames = []
    for s in states:
        s = np.asarray(s)
        N = s.ndim - 1
        z = s.shape[N - 1] // 2
        idx = (slice(None),) * (N - 1) + (z, slice(None))
        sl = s[idx]  # spatial[..] + (W,)
        frame = np.concatenate([sl[..., b] for b in range(sl.shape[-1])], axis=1)
        frame = np.kron(frame.astype(np.float32), np.ones((4, 4), np.float32))
        frames.append((frame * 255).astype(np.uint8))
    try:
        imageio.mimsave(str(path), frames, duration=1.0 / fps)
    except TypeError:
        imageio.mimsave(str(path), frames)
    return True
