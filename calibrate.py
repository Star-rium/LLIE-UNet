"""
calibrate_metrics.py
Thử các biến thể định nghĩa MLI/CI/E/AG và so với bảng baseline của ban tổ chức.

Cách dùng:
  python calibrate_metrics.py selftest
  python calibrate_metrics.py run --low <thư_mục_Low> [--gt <thư_mục_Normal>] [--seeds 42]
  python calibrate_metrics.py run --low <Low> --seeds 1-30        # trung bình nhiều seed

Chỉ cần numpy, Pillow, opencv-python.
"""
import argparse, itertools, random, sys
from pathlib import Path
import numpy as np
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp"}

# ---- Bảng của ban tổ chức (mean) ----
TARGET = {
    "Input": dict(MLI=0.0500, CI=0.0443, E=4.5533, AG=0.0129),
    "CLAHE": dict(MLI=0.1948, CI=0.1273, E=6.6399, AG=0.0647),
    "AWF":   dict(MLI=0.3317, CI=0.1963, E=7.2184, AG=0.1052),
    "AGCWD": dict(MLI=0.2259, CI=0.2085, E=6.8703, AG=0.0793),
    "SDD":   dict(MLI=0.2425, CI=0.1192, E=6.5764, AG=0.0240),
    "RRM":   dict(MLI=0.2724, CI=0.0922, E=6.2910, AG=0.0207),
}
SD_INPUT = dict(MLI=0.0314, CI=0.0286, E=0.7764, AG=0.0067)

# ---- Biến thể ảnh: img là float (H,W,3) thang [0,1] ----
def to_gray(img):
    return img @ np.array([0.299, 0.587, 0.114])

def planes(img, mode):
    """Trả về list các mặt phẳng 2D để tính metric."""
    if mode == "gray":
        return [to_gray(img)]
    if mode == "rgb_mean":   # tính từng kênh rồi lấy trung bình
        return [img[..., c] for c in range(3)]
    if mode == "max":        # kênh V trong HSV
        return [img.max(axis=2)]
    raise ValueError(mode)

# ---- MLI ----
def mli(p):
    return p.mean()

# ---- CI: các ứng viên ----
def ci_std(p):      return p.std()
def ci_michelson(p): return (p.max() - p.min()) / (p.max() + p.min() + 1e-12)
def ci_range(p):    return p.max() - p.min()
def ci_p99_p1(p):   return np.percentile(p, 99) - np.percentile(p, 1)
CI_VARIANTS = {"std": ci_std, "michelson": ci_michelson,
               "range": ci_range, "p99-p1": ci_p99_p1}

# ---- Entropy ----
def entropy(p):
    q = np.clip(np.round(p * 255), 0, 255).astype(np.int64)
    h = np.bincount(q.ravel(), minlength=256).astype(np.float64)
    pr = h / h.sum()
    pr = pr[pr > 0]
    return float(-(pr * np.log2(pr)).sum())

# ---- AG: các ứng viên ----
def ag_fwd_div2(p):   # sqrt((gx^2+gy^2)/2), sai phân tiến
    gx = p[:-1, :-1] - p[:-1, 1:]; gy = p[:-1, :-1] - p[1:, :-1]
    return np.sqrt((gx**2 + gy**2) / 2).mean()
def ag_fwd_nodiv(p):  # sqrt(gx^2+gy^2)
    gx = p[:-1, :-1] - p[:-1, 1:]; gy = p[:-1, :-1] - p[1:, :-1]
    return np.sqrt(gx**2 + gy**2).mean()
def ag_np_gradient(p):  # np.gradient (sai phân trung tâm)
    gy, gx = np.gradient(p)
    return np.sqrt((gx**2 + gy**2) / 2).mean()
def ag_sobel(p):
    import cv2
    gx = cv2.Sobel(p.astype(np.float64), cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(p.astype(np.float64), cv2.CV_64F, 0, 1, ksize=3)
    return np.sqrt((gx**2 + gy**2) / 2).mean()
AG_VARIANTS = {"fwd/2": ag_fwd_div2, "fwd": ag_fwd_nodiv,
               "np.grad": ag_np_gradient, "sobel": ag_sobel}

def avg_over(fn, ps):
    return float(np.mean([fn(p) for p in ps]))

def compute(img, mode, ci_name, ag_name):
    ps = planes(img, mode)
    return dict(MLI=avg_over(mli, ps),
                CI=avg_over(CI_VARIANTS[ci_name], ps),
                E=avg_over(entropy, ps),
                AG=avg_over(AG_VARIANTS[ag_name], ps))

# ---- Dữ liệu ----
def list_images(d):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)

def load(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0

def pick(files, n, seed):
    names = [f.name for f in files]
    chosen = set(random.Random(seed).sample(names, n))
    return [f for f in files if f.name in chosen]

def clahe(img, clip=2.0, grid=8):
    """CLAHE trên kênh L của LAB (cách phổ biến nhất)."""
    import cv2
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    lab[..., 0] = c.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB).astype(np.float64) / 255.0

def clahe_gray_only(img, clip=2.0, grid=8):
    """CLAHE trên từng kênh RGB (biến thể thứ hai)."""
    import cv2
    u8 = np.clip(img * 255, 0, 255).astype(np.uint8)
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    out = np.stack([c.apply(u8[..., k]) for k in range(3)], axis=2)
    return out.astype(np.float64) / 255.0

# ---- Tự kiểm tra ----
def selftest():
    ok = True
    def check(name, cond, info=""):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name, info)
        ok &= bool(cond)
    rng = np.random.default_rng(0)
    black = np.zeros((64, 64, 3)); gray = np.full((64, 64, 3), 0.5)
    noise = rng.random((64, 64, 3))
    ramp = np.tile(np.linspace(0, 1, 64), (64, 1))[..., None].repeat(3, 2)
    for mode in ["gray", "rgb_mean"]:
        m = lambda im: compute(im, mode, "std", "fwd/2")
        b, g, n = m(black), m(gray), m(noise)
        check(f"[{mode}] ảnh đen: MLI=0,E=0,AG=0", b["MLI"] == 0 and b["E"] == 0 and b["AG"] == 0)
        check(f"[{mode}] ảnh xám đều: E=0,AG=0,CI=0", g["E"] == 0 and g["AG"] == 0 and g["CI"] == 0)
        check(f"[{mode}] noise: E>7.5, AG>0.2", n["E"] > 7.5 and n["AG"] > 0.2,
              f"E={n['E']:.2f} AG={n['AG']:.3f}")
    # Gradient tuyến tính độ dốc 1/63 theo x: AG = (1/63)/sqrt(2)
    r = compute(ramp, "gray", "std", "fwd/2")
    check("AG của ramp khớp công thức", abs(r["AG"] - (1/63)/np.sqrt(2)) < 1e-6, f"{r['AG']:.6f}")
    check("E của ramp ≈ log2(64)=6", abs(r["E"] - 6.0) < 1e-6, f"{r['E']:.4f}")
    # Ảnh tối giả lập: đáng ra khớp xu hướng của dòng Input
    dark = (rng.random((128, 128, 3)) ** 3) * 0.15
    d = compute(dark, "gray", "std", "fwd/2"); l = compute(clahe(dark), "gray", "std", "fwd/2")
    check("CLAHE làm tăng MLI, CI, E, AG trên ảnh tối",
          all(l[k] > d[k] for k in d), f"{ {k: (round(d[k],4), round(l[k],4)) for k in d} }")
    print("\nKẾT QUẢ:", "TẤT CẢ ĐẠT" if ok else "CÓ LỖI")
    return ok

# ---- Chạy calibrate ----
def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-"); return list(range(int(a), int(b) + 1))
    return [int(x) for x in s.split(",")]

def run(args):
    files = list_images(args.low)
    print(f"{len(files)} ảnh trong {args.low}")
    seeds = parse_seeds(args.seeds)
    combos = list(itertools.product(["gray", "rgb_mean", "max"], CI_VARIANTS, AG_VARIANTS))

    # Cache ảnh để khỏi đọc lại
    cache = {}
    def get(f):
        if f not in cache: cache[f] = load(f)
        return cache[f]

    def mean_table(proc=lambda x: x):
        res = {c: [] for c in combos}
        for sd in seeds:
            sel = pick(files, args.n, sd)
            per = {c: [] for c in combos}
            for f in sel:
                img = proc(get(f))
                for c in combos:
                    per[c].append(compute(img, *c))
            for c in combos:
                res[c].append({k: np.mean([r[k] for r in per[c]]) for k in per[c][0]})
        return {c: {k: float(np.mean([r[k] for r in res[c]])) for k in res[c][0]} for c in combos}

    def score(vals, tgt):
        # sai số tương đối trung bình trên 4 chỉ số
        return float(np.mean([abs(vals[k] - tgt[k]) / tgt[k] for k in tgt]))

    print(f"\n=== Bước 1: Input vs dòng Input (seeds={seeds[:3]}{'...' if len(seeds)>3 else ''}) ===")
    t_in = mean_table()
    ranked = sorted(combos, key=lambda c: score(t_in[c], TARGET["Input"]))
    print(f"{'mode':9s}{'CI':10s}{'AG':9s}{'MLI':>8s}{'CI':>8s}{'E':>8s}{'AG':>8s}{'err%':>8s}")
    for c in ranked[:8]:
        v = t_in[c]
        print(f"{c[0]:9s}{c[1]:10s}{c[2]:9s}{v['MLI']:8.4f}{v['CI']:8.4f}{v['E']:8.4f}{v['AG']:8.4f}"
              f"{100*score(v, TARGET['Input']):8.1f}")
    print("mục tiêu :", TARGET["Input"])
    best = ranked[0]
    print("-> Tổ hợp tốt nhất:", best)

    print("\n=== Bước 2: CLAHE (2 biến thể) với tổ hợp tốt nhất vs dòng CLAHE ===")
    for name, fn in [("CLAHE-LAB", clahe), ("CLAHE-RGB", clahe_gray_only)]:
        sel_vals = []
        for sd in seeds:
            sel = pick(files, args.n, sd)
            sel_vals.append([compute(fn(get(f)), *best) for f in sel])
        v = {k: float(np.mean([r[k] for s in sel_vals for r in s])) for k in TARGET["CLAHE"]}
        print(name, {k: round(x, 4) for k, x in v.items()},
              f"err={100*score(v, TARGET['CLAHE']):.1f}%")
    print("mục tiêu :", TARGET["CLAHE"])

    if args.gt:
        print("\n=== Bước 3: GT (Normal) – xem trần của train theo GT so với AWF ===")
        gfiles = {p.name: p for p in list_images(args.gt)}
        vals = []
        for sd in seeds:
            for f in pick(files, args.n, sd):
                g = gfiles.get(f.name.replace("low", "normal")) or gfiles.get(f.name)
                if g is None: continue
                vals.append(compute(load(g), *best))
        if vals:
            v = {k: float(np.mean([r[k] for r in vals])) for k in TARGET["AWF"]}
            print("GT :", {k: round(x, 4) for k, x in v.items()})
            print("AWF:", TARGET["AWF"])
            for k in v:
                print(f"  {k}: GT {'>' if v[k] > TARGET['AWF'][k] else '<'} AWF")
        else:
            print("Không ghép được tên file Low/Normal, kiểm tra lại quy tắc đặt tên.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    r = sub.add_parser("run")
    r.add_argument("--low", required=True)
    r.add_argument("--gt")
    r.add_argument("--n", type=int, default=100)
    r.add_argument("--seeds", default="42")
    a = ap.parse_args()
    if a.cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    run(a)