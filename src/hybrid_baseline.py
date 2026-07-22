"""
hybrid_baseline.py

Baseline correction "hybrid DL": 1 mạng CNN nhỏ (PyTorch) phân loại họ
baseline (đa thức bậc 2 / Gaussian / cả hai / không có) đang trội trong
phổ, sau đó fit tham số bằng scipy.curve_fit theo đúng họ được chọn, rồi
tinh chỉnh phần dư bằng airPLS (TÁI DÙNG raman_processing.baseline_correction
đã có sẵn trong repo -- không viết thêm 1 bản airPLS thứ hai với cách chọn
lambda khác, để tránh 2 nguồn "sự thật" mâu thuẫn nhau trong cùng pipeline).

KHÁC VỚI code tham khảo (dự án anh khóa trên):
- Chuyển từ Keras/TensorFlow sang PyTorch (khớp models/baseline_classifier.pt
  và khớp các model khác trong 06_model_comparison.ipynb).
- Không có phần mã hoá GADF/ảnh 2D -- pipeline này chỉ làm việc trên phổ 1D.
- Sửa 1 lỗi nhỏ ở kiến trúc gốc: output Dense(2, sigmoid) nhưng lại train
  bằng sparse_categorical_crossentropy (không khớp toán học). Ở đây dùng
  đúng cặp sigmoid + BCEWithLogitsLoss cho bài toán multi-label 2 lớp
  (poly có mặt? / gaussian có mặt? -- độc lập nhau, 1 phổ có thể có cả 2).
- input_length THAM SỐ HOÁ theo phổ thực tế của dự án (252 điểm), không
  hard-code 880 như code gốc.

Cách dùng:
    from hybrid_baseline import (BaselineClassifier, load_baseline_classifier,
                                  hybrid_baseline_correction)

    model, meta = load_baseline_classifier('../models/baseline_classifier.pt')
    corrected = hybrid_baseline_correction(spectrum, model, x=x,
                                            airpls_lam=AIRPLS_LAM)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import curve_fit

from raman_processing import baseline_correction


def normalize_minmax(spectrum: np.ndarray) -> np.ndarray:
    """Chuẩn hoá min-max về [0,1]. Dùng NHẤT QUÁN cả lúc train classifier
    (notebook 01) lẫn lúc inference (hybrid_baseline_correction) -- model
    chỉ học đúng nếu thấy cùng 1 thang giá trị ở cả 2 giai đoạn.

    LƯU Ý: đặt tên khác raman_processing.normalize_spectrum() có chủ đích
    -- 2 hàm đó dùng công thức khác hẳn nhau (L2-norm/max/area vs min-max),
    trùng tên dễ gây import shadow âm thầm nếu 1 notebook nào đó import cả
    2 module bằng `from ... import *`."""
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    spectrum = spectrum - np.min(spectrum)
    m = np.max(spectrum)
    if m > 0:
        spectrum = spectrum / m
    return spectrum


# ---------------------------------------------------------------------------
# 1. Các dạng baseline tổng hợp (poly / gaussian / cả hai) -- port nguyên
#    vẹn từ code tham khảo, thuật toán không đổi, chỉ đổi tên biến cho rõ.
# ---------------------------------------------------------------------------
def poly_baseline(x, p, intensity, b):
    """Baseline dạng đa thức bậc p (chuẩn hoá theo len(x))."""
    y = (x / len(x)) ** p + b
    return y * intensity / max(y)


def gaussian_baseline(x, mean, sd, intensity, b):
    """Baseline dạng Gaussian (mô phỏng fluorescence dạng chuông)."""
    y = np.exp(-(x - mean) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)) + b
    return y * intensity / max(y)


def pg_baseline(x, p, in1, mean, sd, in2, b):
    """Kết hợp cả 2 dạng poly + gaussian, mỗi dạng có cường độ riêng."""
    y1 = (x / len(x)) ** p + b
    y2 = np.exp(-(x - mean) ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)) + b
    return y1 / max(y1) * in1 + y2 / max(y2) * in2


def mix_min_no(sp, baseline):
    """Lấy min theo từng điểm giữa phổ và baseline ước lượng (đảm bảo
    baseline không vượt lên trên phổ thật)."""
    return np.minimum(baseline, sp)


# ---------------------------------------------------------------------------
# 2. Mạng phân loại họ baseline (PyTorch)
# ---------------------------------------------------------------------------
class BaselineClassifier(nn.Module):
    """CNN 1D nhỏ: input 1 phổ (length=input_length) -> 2 logit
    (poly_present, gaussian_present). Dùng sigmoid ở bước inference,
    train bằng BCEWithLogitsLoss (logit thô, không sigmoid trong forward).

    Kiến trúc giữ tinh thần bản gốc (1 Conv1D + AveragePooling + Dense),
    chỉ tham số hoá input_length thay vì hard-code 880.
    """

    def __init__(self, input_length: int, conv_filters: int = 16, kernel_size: int = 5):
        super().__init__()
        self.input_length = input_length
        self.conv = nn.Conv1d(1, conv_filters, kernel_size=kernel_size, stride=1)
        self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        conv_out_len = input_length - kernel_size + 1  # 'valid' conv, stride 1
        pool_out_len = conv_out_len // 2
        flat_dim = conv_filters * pool_out_len

        self.fc1 = nn.Linear(flat_dim, 100)
        self.fc2 = nn.Linear(100, 2)  # logits: [poly_present, gaussian_present]

    def forward(self, x):
        # x: (batch, input_length) -> (batch, 1, input_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.conv(x))
        x = self.pool(x)
        x = x.flatten(start_dim=1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)  # logits thô, KHÔNG sigmoid ở đây


def load_baseline_classifier(path: str, device: str = "cpu"):
    """Load model + metadata (input_length) đã lưu bằng save_baseline_classifier."""
    ckpt = torch.load(path, map_location=device)
    model = BaselineClassifier(input_length=ckpt["input_length"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, {"input_length": ckpt["input_length"]}


def save_baseline_classifier(model: BaselineClassifier, path: str):
    torch.save({"state_dict": model.state_dict(), "input_length": model.input_length}, path)


# ---------------------------------------------------------------------------
# 3. Hybrid baseline correction (inference time)
# ---------------------------------------------------------------------------
def _predict_family(model: BaselineClassifier, spectrum: np.ndarray, device: str = "cpu"):
    """Trả về (poly_prob, gaussian_prob) qua sigmoid(logits)."""
    with torch.no_grad():
        x = torch.tensor(spectrum, dtype=torch.float32, device=device).unsqueeze(0)
        logits = model(x)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])


def iterative_fitting_with_bounds(spectrum, model, device: str = "cpu", ite: int = 10):
    """Lặp lại: hỏi model 'phổ còn baseline dạng nào', fit tham số bằng
    curve_fit theo đúng dạng đó, trừ dần khỏi phổ (mix_min_no), lặp lại
    tối đa `ite` lần. Port từ code tham khảo, chỉ đổi tf.Module -> torch.

    Nếu model không tự tin về dạng nào cả (2 xác suất đều < 0.5), dừng sớm
    -- không có trong code gốc, thêm vào để tránh lặp vô ích khi không còn
    baseline nào đáng kể (bản gốc chạy đủ `ite` vòng ngay cả khi không cần).
    """
    x = np.linspace(1, spectrum.shape[0], spectrum.shape[0])
    tempb = spectrum.copy()

    for i in range(ite):
        poly_p, gauss_p = _predict_family(model, tempb, device=device)

        if poly_p < 0.5 and gauss_p < 0.5:
            break  # model cho rằng không còn baseline đáng kể -> dừng sớm

        fitted_baseline = tempb  # fallback nếu curve_fit thất bại
        try:
            if poly_p >= 0.5 and gauss_p >= 0.5:
                p, _ = curve_fit(
                    pg_baseline, x, tempb,
                    bounds=([1, 0.5, 0, 100, 0.5, -0.5],
                            [3, 1, spectrum.shape[0], 600, 1, 0.5]),
                    maxfev=10000)
                fitted_baseline = pg_baseline(x, *p)
            elif poly_p >= 0.5:
                p, _ = curve_fit(
                    poly_baseline, x, tempb,
                    bounds=([1, 0.5, -0.5], [3, 1, 0.5]),
                    maxfev=10000)
                fitted_baseline = poly_baseline(x, *p)
            elif gauss_p >= 0.5:
                p, _ = curve_fit(
                    gaussian_baseline, x, tempb,
                    bounds=([0, 100, 0.5, -0.5], [spectrum.shape[0], 600, 1, 0.5]),
                    maxfev=10000)
                fitted_baseline = gaussian_baseline(x, *p)
        except RuntimeError:
            pass  # giữ fitted_baseline = tempb (không sửa gì vòng này)

        tempb = mix_min_no(tempb, fitted_baseline)

    return tempb


def hybrid_baseline_correction(spectrum, model, device: str = "cpu",
                                airpls_lam: float = 1e5, airpls_p: float = 0.01,
                                airpls_niter: int = 10, ite: int = 10,
                                clip_negative: bool = True):
    """Baseline correction 2 giai đoạn, đúng công thức trong code tham khảo:

        coarse_baseline = iterative_fitting_with_bounds(spectrum, model)
        fine_baseline    = airPLS(coarse_baseline)   # làm mượt thêm 1 lần nữa
        corrected        = spectrum - fine_baseline

    Bước airPLS thứ 2 TÁI DÙNG raman_processing.baseline_correction() sẵn có
    trong repo (qua return_baseline=True để lấy đúng phần baseline, không
    phải phần đã trừ) -- không viết thêm airPLS riêng, tránh 2 nguồn "sự
    thật" khác nhau về cách chọn lambda trong cùng pipeline.

    airpls_lam/p/niter : NÊN lấy từ chosen_params.json (AIRPLS_LAM) để nhất
               quán với phần còn lại của pipeline (00/02).

    Nếu bước DL-guided lỗi vì bất kỳ lý do gì, fallback về airPLS thuần
    (giống code tham khảo).
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    try:
        coarse_baseline = iterative_fitting_with_bounds(spectrum, model, device=device, ite=ite)
        _, fine_baseline = baseline_correction(
            coarse_baseline, method="airpls", lam=airpls_lam, p=airpls_p,
            niter=airpls_niter, return_baseline=True)
        corrected = spectrum - fine_baseline
    except Exception as e:  # noqa: BLE001 -- fallback có chủ đích, giống code gốc
        print(f"[hybrid_baseline_correction] Lỗi ở bước DL-guided: {e}. Fallback airPLS thuần.")
        corrected = baseline_correction(spectrum, method="airpls", lam=airpls_lam,
                                        p=airpls_p, niter=airpls_niter)

    if clip_negative:
        corrected = np.clip(corrected, 0, None)
    return corrected