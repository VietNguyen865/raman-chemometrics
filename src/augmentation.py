"""
augmentation.py

Tăng cường dữ liệu (data augmentation) cho phổ Raman, theo mô tả mục 3.2.2:
biến đổi từng phổ gốc ĐỘC LẬP (không nội suy/trộn giữa các phổ khác nhau,
để tránh tạo đặc trưng giả/phi vật lý).

6 kỹ thuật áp dụng cho mỗi phổ tổng hợp:
    1. Nhiễu Gaussian       : sigma = 0.05 * std(phổ) * scale (scale ~ U[1.0, 2.0])
    2. Trôi nền đa thức bậc 2: xác suất 30%, dạng a*x^2 + b*x + c
    3. Nhiễu giả huỳnh quang : suy giảm mũ, dạng amp * exp(-lambda * (x - x_min))
    4. Dịch phổ             : dịch toàn bộ phổ +- vài điểm dọc trục số sóng
    5. Dãn/nén phổ          : hệ số +-1-2%
    6. Biến đổi cường độ    : nhân toàn phổ với hệ số ngẫu nhiên U[0.8, 1.2]
"""
import numpy as np


def augment_spectrum(spectrum, x, n_augmentations=100, seed=None,
                      noise_std_frac=0.05, noise_scale_range=(1.0, 2.0),
                      baseline_drift_prob=0.3, baseline_drift_frac=0.10,
                      fluor_lambda=0.01, fluor_lambda_jitter=0.5,
                      fluor_amp_frac=0.3,
                      shift_max_points=3,
                      stretch_range=(-0.02, 0.02),
                      intensity_range=(0.8, 1.2)):
    """
    Sinh n_augmentations phổ tổng hợp từ 1 phổ gốc, theo đúng thứ tự 6 bước
    trong mục 3.2.2. Tất cả phổ tổng hợp được resample về CÙNG trục x với
    phổ gốc (để tiện ghép bảng đặc trưng/so sánh giữa các mẫu về sau).

    spectrum : mảng cường độ phổ gốc (1D).
    x        : trục Raman shift (cm-1), cùng độ dài với spectrum.
    n_augmentations : số phổ tổng hợp sinh ra từ phổ gốc này.
    seed     : cố định random seed để tái lập kết quả.

    noise_std_frac      : hệ số std nhiễu Gaussian = noise_std_frac * std(phổ).
    noise_scale_range    : khoảng ngẫu nhiên nhân thêm vào std nhiễu (đa dạng hoá).
    baseline_drift_prob  : xác suất áp dụng trôi nền đa thức bậc 2.
    baseline_drift_frac  : biên độ trôi nền tối đa, tính theo tỉ lệ của
                            (max(phổ) - min(phổ)) -- dùng tỉ lệ thay vì số
                            tuyệt đối vì biên độ phổ chênh lệch rất lớn giữa
                            các mẫu (vd Ethanol nguyên chất ~5600 vs E5 ~287).
    fluor_lambda         : hệ số suy giảm mũ cơ sở (xấp xỉ 0.01 theo mô tả).
    fluor_lambda_jitter   : biên độ dao động ngẫu nhiên quanh fluor_lambda
                            (lambda thực tế ~ U[(1-jitter), (1+jitter)] * fluor_lambda).
    fluor_amp_frac        : biên độ cực đại của thành phần giả huỳnh quang,
                            tính theo tỉ lệ max(phổ).
    shift_max_points      : dịch phổ tối đa +-shift_max_points điểm mẫu.
    stretch_range          : hệ số dãn/nén trục x, mặc định +-2%.
    intensity_range         : hệ số nhân cường độ toàn phổ.

    Trả về: mảng shape (n_augmentations, len(spectrum)).
    """
    rng = np.random.default_rng(seed)
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    n = len(spectrum)

    spec_std = np.std(spectrum)
    spec_ptp = np.ptp(spectrum)
    spec_max = np.max(spectrum)

    outputs = np.empty((n_augmentations, n), dtype=float)

    for i in range(n_augmentations):
        s = spectrum.copy()

        # --- 1. Nhiễu Gaussian ---
        scale = rng.uniform(*noise_scale_range)
        sigma = noise_std_frac * spec_std * scale
        s = s + rng.normal(0.0, sigma, size=n)

        # --- 2. Trôi nền đa thức bậc 2 (xác suất 30%) ---
        if rng.random() < baseline_drift_prob:
            u = (x - x.min()) / (x.max() - x.min()) * 2 - 1  # chuẩn hoá x -> [-1,1]
            a, b, c = rng.normal(0.0, 1.0, size=3)
            drift = a * u**2 + b * u + c
            drift = drift / (np.ptp(drift) + 1e-12)  # chuẩn hoá biên độ về ~1
            target_amp = rng.uniform(0.0, baseline_drift_frac) * spec_ptp
            s = s + drift * target_amp

        # --- 3. Nhiễu giả huỳnh quang (suy giảm mũ) ---
        lam = fluor_lambda * rng.uniform(1 - fluor_lambda_jitter, 1 + fluor_lambda_jitter)
        amp = rng.uniform(0.0, fluor_amp_frac) * spec_max
        fluor = amp * np.exp(-lam * (x - x.min()))
        s = s + fluor

        # --- 4. Dịch phổ +- vài điểm ---
        shift = int(rng.integers(-shift_max_points, shift_max_points + 1))
        if shift != 0:
            s = np.roll(s, shift)
            # lặp giá trị biên thay vì để wrap-around (tránh giả tạo đỉnh ở đầu/cuối)
            if shift > 0:
                s[:shift] = s[shift]
            else:
                s[shift:] = s[shift - 1]

        # --- 5. Dãn/nén phổ +-1-2% ---
        factor = 1.0 + rng.uniform(*stretch_range)
        x_stretched = x[0] + (x - x[0]) * factor
        s = np.interp(x, x_stretched, s)  # resample lại về đúng trục x gốc

        # --- 6. Biến đổi cường độ (công suất laser) ---
        intensity_factor = rng.uniform(*intensity_range)
        s = s * intensity_factor

        outputs[i] = s

    return outputs


def augment_dataset(df, x, sample_cols=None, n_augmentations=100, seed=None, **kwargs):
    """
    Áp dụng augment_spectrum() cho toàn bộ các cột mẫu trong DataFrame.

    Trả về:
        augmented_df : DataFrame, mỗi cột là 1 phổ tổng hợp, tên cột dạng
                        '{tên_mẫu_gốc}__aug{i}'.
        source_map   : dict {tên_cột_augmented: tên_mẫu_gốc}, dùng để giữ
                        liên kết nhãn (nồng độ, series...) khi ghép với
                        data_cleaning.parse_sample_label().
    """
    import pandas as pd

    if sample_cols is None:
        sample_cols = [c for c in df.columns if c != "Raman Shift (cm-1)"]

    x = np.asarray(x, dtype=float)
    augmented_data = {}
    source_map = {}

    for col in sample_cols:
        spectrum = df[col].values.astype(float)
        # seed riêng theo tên cột để tái lập được nhưng vẫn khác nhau giữa các mẫu
        col_seed = None if seed is None else seed + abs(hash(col)) % (2**16)
        synth = augment_spectrum(spectrum, x, n_augmentations=n_augmentations,
                                  seed=col_seed, **kwargs)
        for i in range(n_augmentations):
            new_col = f"{col}__aug{i}"
            augmented_data[new_col] = synth[i]
            source_map[new_col] = col

    augmented_df = pd.DataFrame(augmented_data)
    return augmented_df, source_map