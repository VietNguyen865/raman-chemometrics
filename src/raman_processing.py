import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import medfilt, savgol_filter, find_peaks
from lmfit.models import VoigtModel, GaussianModel, LorentzianModel


# ---------------------------------------------------------------------------
# 1. Cosmic ray removal
# ---------------------------------------------------------------------------
def remove_cosmic_rays(spectrum, window=5, threshold=7):
    """
    Lọc cosmic ray spikes bằng median filter.

    window: kích thước cửa sổ median (số lẻ), nên nhỏ hơn độ rộng đỉnh Raman thật
    threshold: số lần MAD (median absolute deviation) để coi là spike
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    if window % 2 == 0:
        window += 1
    med = medfilt(spectrum, kernel_size=window)
    residual = spectrum - med

    mad = np.median(np.abs(residual - np.median(residual)))
    if mad == 0:
        mad = 1e-9

    spike_mask = np.abs(residual) > threshold * mad

    cleaned = spectrum.copy()
    cleaned[spike_mask] = med[spike_mask]

    return cleaned, spike_mask


# ---------------------------------------------------------------------------
# 2. Baseline correction (fluorescence background)
# ---------------------------------------------------------------------------
def baseline_correction(spectrum, method='airpls', lam=1e5, p=0.01, niter=10,
                         cutoff_ratio=0.02, taper_ratio=0.25, pad_ratio=0.5,
                         return_baseline=False):
    """
    Loại fluorescence background khỏi phổ Raman.

    method: 'airpls' (mặc định), 'als', hoặc 'fft'

    LƯU Ý CHỌN METHOD (rút ra từ thực nghiệm trên phổ ethanol/methanol,
    N~250 điểm, dx~11 cm-1):
    - 'airpls': mặc định khuyến nghị. Hoạt động trong miền giá trị thực,
      dùng trọng số bất đối xứng nên không phụ thuộc vào việc tách baseline
      và đỉnh theo tần số -> ổn định với phổ ngắn/đỉnh tương đối rộng.
      Hạn chế đã biết (Vu Duong et al., 2024, VJSTE): kém chính xác hơn ở
      vùng số sóng thấp so với vùng cao.
    - 'fft': KHÔNG khuyến nghị cho phổ ít điểm (N nhỏ) có đỉnh rộng so với
      tổng dải đo. Khi đó bao hình đỉnh và baseline chồng lấn trong miền
      tần số thấp, khiến baseline bị "hút" theo đỉnh cao, gây corrected
      spectrum âm sâu bất thường giữa các đỉnh. Chỉ phù hợp khi có sự phân
      tách tần số rõ ràng (phổ độ phân giải cao, đỉnh hẹp so với N).

    Tham số riêng cho 'fft':
        cutoff_ratio : tỉ lệ (0-1) xác định biên tần số thấp được coi là baseline.
                       Giá trị nhỏ -> baseline mượt hơn nhưng có thể giữ lại
                       sóng nền lởm chởm. Giá trị lớn -> dễ ăn vào chân đỉnh rộng.
                       Thường thử trong khoảng 0.005 - 0.05, tùy độ rộng đỉnh
                       và tổng số điểm phổ.
        taper_ratio  : tỉ lệ cửa sổ chuyển tiếp mượt (cosine taper) ngay tại
                       biên cutoff để giảm hiệu ứng Gibbs/ringing.
        pad_ratio    : tỉ lệ mirror-padding hai đầu phổ để giảm edge artifact
                       do FFT giả định tín hiệu tuần hoàn.
        return_baseline: nếu True, trả về thêm mảng baseline đã ước lượng để
                       kiểm tra/visualize.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()

    if method == 'fft':
        corrected, baseline = _fft_baseline(
            spectrum,
            cutoff_ratio=cutoff_ratio,
            taper_ratio=taper_ratio,
            pad_ratio=pad_ratio,
        )
    elif method == 'airpls':
        baseline = spectrum - _airpls(spectrum, lam=lam, p=p, niter=niter)
        corrected = spectrum - baseline
    elif method == 'als':
        baseline = spectrum - _als(spectrum, lam=lam, p=p, niter=niter)
        corrected = spectrum - baseline
    else:
        raise ValueError(f"Unknown method: {method}")

    if return_baseline:
        return corrected, baseline
    return corrected


def _fft_baseline(spectrum, cutoff_ratio=0.02, taper_ratio=0.25, pad_ratio=0.5):
    """
    Ước lượng baseline bằng low-pass filter trong miền tần số (FFT).

    Ý tưởng: fluorescence background biến thiên chậm -> nằm ở tần số thấp.
    Đỉnh Raman hẹp -> nằm ở tần số cao hơn. Giữ lại thành phần tần số thấp
    (có taper mượt để tránh ringing) rồi biến đổi ngược để có baseline,
    sau đó trừ khỏi phổ gốc.
    """
    n = len(spectrum)

    # --- mirror padding để giảm edge artifact ---
    pad = max(1, int(n * pad_ratio))
    pad = min(pad, n - 1)
    left_pad = spectrum[pad:0:-1]
    right_pad = spectrum[-2:-pad - 2:-1]
    padded = np.concatenate([left_pad, spectrum, right_pad])
    N = len(padded)

    # --- FFT ---
    spec_fft = np.fft.rfft(padded)
    n_freq = len(spec_fft)

    cutoff_idx = max(1, int(cutoff_ratio * N))
    cutoff_idx = min(cutoff_idx, n_freq)

    baseline_fft = np.zeros_like(spec_fft)
    baseline_fft[:cutoff_idx] = spec_fft[:cutoff_idx]

    # --- taper mượt ở biên cutoff để giảm Gibbs ringing ---
    taper_width = max(1, int(cutoff_idx * taper_ratio))
    taper_width = min(taper_width, cutoff_idx)
    if taper_width > 1:
        ramp = 0.5 * (1 + np.cos(np.linspace(0, np.pi, taper_width)))
        baseline_fft[cutoff_idx - taper_width:cutoff_idx] *= ramp

    baseline_padded = np.fft.irfft(baseline_fft, n=N)
    baseline = baseline_padded[pad:pad + n]

    corrected = spectrum - baseline
    return corrected, baseline


def _airpls(spectrum, lam, p, niter):
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = spectrum.copy()

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * spectrum)
        d = spectrum - z
        dn = d[d < 0]
        if len(dn) == 0:
            break
        m = np.mean(dn)
        s = np.std(dn)
        if s == 0:
            break
        w = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

    return spectrum - z


def _als(spectrum, lam, p, niter):
    L = len(spectrum)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2), dtype=float)
    D = lam * D.dot(D.transpose())
    w = np.ones(L)
    z = spectrum.copy()

    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = (W + D).tocsc()
        z = spsolve(Z, w * spectrum)
        w = p * (spectrum > z) + (1 - p) * (spectrum < z)

    return spectrum - z


def normalize_spectrum(spectrum, method='l2'):
    """
    Loại bỏ chênh lệch scale cường độ giữa các lần đo (laser power/thời
    gian tích phân khác nhau) -- áp dụng SAU baseline_correction(), TRƯỚC
    khi đưa vào augmentation/PCA/model.

    method:
        'l2'  : chia cho norm-2 của toàn phổ (khuyến nghị mặc định)
        'max' : chia cho giá trị cực đại
        'area': chia cho tổng cường độ (diện tích)

    ⚠️ method='area' trên phổ ĐÃ baseline_correction() có thể chia cho số
    gần 0 hoặc ÂM: sau khi trừ baseline, phổ dao động quanh 0 ở vùng nền,
    và với mẫu SNR thấp (vd E5_a, EM8_c -- xem snr_report.csv) phần âm của
    nhiễu nền có thể lớn hơn phần dương của đỉnh, khiến spectrum.sum() gần
    0/âm. Khi đó phổ bị lật dấu hoặc phóng đại bất thường mà KHÔNG có
    exception nào -- lỗi âm thầm. Hàm này cảnh báo rõ khi rơi vào trường
    hợp đó, tương tự cách calc_snr() đã cảnh báo cho I_BG<=0.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    if method == 'l2':
        denom = np.linalg.norm(spectrum)
    elif method == 'max':
        denom = spectrum.max()
    elif method == 'area':
        denom = spectrum.sum()
        if denom <= 1e-6:
            import warnings
            warnings.warn(
                f"normalize_spectrum(method='area'): tổng cường độ gần 0 hoặc "
                f"âm (denom={denom:.4g}) -- kết quả chuẩn hoá có thể bị lật dấu/"
                f"phóng đại bất thường. Cân nhắc dùng method='l2' hoặc 'max' "
                f"thay thế, đặc biệt với mẫu SNR thấp.",
                RuntimeWarning,
            )
    else:
        raise ValueError(f"Unknown method: {method}")
    return spectrum / (denom + 1e-9)


# ---------------------------------------------------------------------------
# 3. Smoothing
# ---------------------------------------------------------------------------
def smooth_spectrum(spectrum, window=5, polyorder=3):
    """
    Savitzky-Golay smoothing.

    window mặc định = 5 (không phải 15): window rộng làm mượt tốt hơn
    nhưng có thể GỘP các đỉnh chồng lấn gần nhau thành 1 đỉnh giả (đã xác
    nhận thực nghiệm: đỉnh đôi 1254/1290 cm-1 của ethanol bị savgol
    window=7 gộp làm 1 đỉnh tại 1278). Nếu phổ của bạn không có đỉnh đôi
    sát nhau và cần giảm nhiễu mạnh hơn, có thể tăng window, nhưng cần
    kiểm tra lại bằng mắt vùng đỉnh chồng lấn trước khi dùng.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        polyorder = window - 1
    return savgol_filter(spectrum, window_length=window, polyorder=polyorder)


# ---------------------------------------------------------------------------
# 3b. Noise estimation & adaptive threshold / SNR (dùng chung cho phần 4 và
#     cho đánh giá định lượng khi viết báo cáo/paper)
# ---------------------------------------------------------------------------
def estimate_noise_std(spectrum, x, bg_range=(1900, 2700)):
    """
    Ước lượng độ lệch chuẩn nhiễu nền từ 1 vùng phẳng, không có đỉnh hóa học
    đã biết. Mặc định bg_range=(1900,2700) phù hợp với ethanol/methanol;
    ĐỔI lại range này nếu chất của bạn có đỉnh nằm trong khoảng đó.

    spectrum: phổ đã baseline-corrected (chưa cần smooth).
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    mask = (x >= bg_range[0]) & (x <= bg_range[1])
    if mask.sum() < 5:
        raise ValueError("bg_range quá hẹp hoặc không khớp với trục x của phổ")
    std = np.std(spectrum[mask])
    return std if (std > 0 and not np.isnan(std)) else 1.0


def adaptive_prominence(spectrum, x, bg_range=(1900, 2700), k=4, floor=10.0):
    """
    Ngưỡng prominence = k * noise_std, thay cho số cố định.

    k=4 là điểm cân bằng đã kiểm chứng thực nghiệm trên bộ dữ liệu
    ethanol/methanol (k=3 sinh đỉnh giả trong vùng nhiễu; k=5 bắt đầu mất
    đỉnh thật). Với dữ liệu khác, NÊN kiểm chứng lại k bằng permutation
    test thay vì dùng mặc định này trực tiếp cho paper.

    floor: ngưỡng sàn tối thiểu, đề phòng noise_std ước lượng quá nhỏ do
    mẫu ít điểm/trùng hợp ngẫu nhiên.
    """
    noise_std = estimate_noise_std(spectrum, x, bg_range=bg_range)
    return max(k * noise_std, floor)


def calc_snr(spectrum, x, peak_x, bg_range=(1900, 2700), peak_window=15):
    """
    SNR = (I_peak - I_BG) / I_BG, theo công thức Vu Duong et al. (2024),
    VJSTE, Eq.(1). Dùng để báo cáo định lượng mức cải thiện SNR trước/sau
    baseline correction cho từng đỉnh cụ thể (vd trong phần Results).

    ⚠️ CHỈ dùng hàm này trên phổ RAW (chưa baseline-corrected), hoặc phổ đã
    chuẩn hóa còn dư baseline dương gần 0. TUYỆT ĐỐI KHÔNG gọi hàm này trên
    phổ đã baseline_correction(...) xong -- I_BG khi đó gần 0 hoặc có thể
    ÂM (do airPLS/als fit rất sát, dao động nhiễu quanh 0), khiến công thức
    chia cho số gần 0/âm và cho ra giá trị vô nghĩa (đã xác nhận thực
    nghiệm: SNR tính ra -700 trên phổ ethanol đã airPLS). Với phổ đã
    baseline-corrected, dùng calc_snr_std() thay thế.

    spectrum : phổ RAW cần đánh giá (không phải phổ đã trừ baseline).
    peak_x   : vị trí đỉnh (cm-1) cần tính SNR.
    bg_range : vùng tham chiếu nền, không có đỉnh Raman.
    peak_window: bề rộng (cm-1) quanh peak_x để lấy cường độ trung bình đỉnh.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    peak_mask = (x >= peak_x - peak_window / 2) & (x <= peak_x + peak_window / 2)
    bg_mask = (x >= bg_range[0]) & (x <= bg_range[1])

    if peak_mask.sum() == 0 or bg_mask.sum() == 0:
        raise ValueError("peak_window hoặc bg_range không khớp với trục x")

    i_peak = spectrum[peak_mask].mean()
    i_bg = spectrum[bg_mask].mean()

    if i_bg <= 0:
        import warnings
        warnings.warn(
            "calc_snr(): I_BG <= 0 -- hàm này không ổn định trên phổ đã "
            "baseline-corrected. Dùng calc_snr_std() thay thế.",
            RuntimeWarning,
        )
        return np.nan
    return (i_peak - i_bg) / i_bg


def calc_snr_std(spectrum, x, peak_x, bg_range=(1900, 2700), peak_window=15):
    """
    SNR cổ điển = I_peak / sigma_noise (độ lệch chuẩn nhiễu nền), thay cho
    calc_snr() khi spectrum đã qua baseline_correction(...).

    Lý do cần hàm riêng: sau khi trừ baseline, I_BG (trung bình vùng nền)
    tiến gần 0 và có thể âm do dao động nhiễu -- công thức (I_peak-I_BG)/I_BG
    của calc_snr() khi đó chia cho số gần 0/âm, cho kết quả vô nghĩa.
    Dùng sigma (độ lệch chuẩn) thay vì mean làm mẫu số tránh được vấn đề này.

    spectrum : phổ ĐÃ baseline-corrected.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()

    peak_mask = (x >= peak_x - peak_window / 2) & (x <= peak_x + peak_window / 2)
    bg_mask = (x >= bg_range[0]) & (x <= bg_range[1])

    if peak_mask.sum() == 0 or bg_mask.sum() == 0:
        raise ValueError("peak_window hoặc bg_range không khớp với trục x")

    i_peak = spectrum[peak_mask].mean()
    noise_std = spectrum[bg_mask].std()

    if noise_std == 0 or np.isnan(noise_std):
        return np.nan
    return i_peak / noise_std


# ---------------------------------------------------------------------------
# 4. Peak detection
# ---------------------------------------------------------------------------
def detect_peaks(spectrum, x=None, prominence=10, distance=1, width=0.5):
    """
    Wrapper quanh scipy.find_peaks.

    distance=1, width=0.5 (không phải 5/2 như mặc định cũ): giá trị lớn
    hơn có thể loại nhầm đỉnh đôi chồng lấn sát nhau (case 1254/1290 cm-1
    của ethanol từng bị width=1 loại mất đỉnh 1254). Nếu phổ của bạn không
    có đỉnh sát nhau, có thể tăng lên để giảm đỉnh giả do nhiễu.

    prominence: nên truyền vào kết quả của adaptive_prominence(...) thay
    vì số cố định, trừ khi bạn đã có lý do cụ thể để cố định ngưỡng.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    peak_idx, properties = find_peaks(
        spectrum,
        prominence=prominence,
        distance=distance,
        width=width,
    )

    result = {
        'index': peak_idx,
        'prominence': properties['prominences'],
        'width': properties['widths'],
        'left_base': properties['left_bases'],
        'right_base': properties['right_bases'],
    }

    if x is not None:
        result['position'] = np.asarray(x)[peak_idx]

    return result


# ---------------------------------------------------------------------------
# 5. Peak fitting (Voigt/Gaussian/Lorentzian) — xử lý đỉnh chồng lấn
# ---------------------------------------------------------------------------
def _get_model(profile, prefix):
    if profile == 'voigt':
        return VoigtModel(prefix=prefix)
    elif profile == 'gaussian':
        return GaussianModel(prefix=prefix)
    elif profile == 'lorentzian':
        return LorentzianModel(prefix=prefix)
    else:
        raise ValueError(f"Unknown profile: {profile}")


def fit_peak(spectrum, x, peak_centers, profile='voigt', window=20, sigma_max=30):
    """
    Fit 1 hoặc nhiều đỉnh chồng lấn trong 1 vùng bằng lmfit.
    """
    x = np.asarray(x, dtype=float)
    spectrum = np.asarray(spectrum, dtype=float)
    peak_centers = np.atleast_1d(peak_centers)

    lo = min(peak_centers) - window
    hi = max(peak_centers) + window
    mask = (x >= lo) & (x <= hi)
    x_fit = x[mask]
    y_fit = spectrum[mask]

    model = None
    params = None

    for i, center in enumerate(peak_centers):
        prefix = f'p{i}_'
        comp = _get_model(profile, prefix)
        comp_params = comp.make_params()
        comp_params[f'{prefix}center'].set(value=center, min=center - 5, max=center + 5)
        comp_params[f'{prefix}amplitude'].set(value=max(y_fit.max(), 1e-3), min=0)
        comp_params[f'{prefix}sigma'].set(value=3, min=0.5, max=sigma_max)

        if model is None:
            model = comp
            params = comp_params
        else:
            model = model + comp
            params.update(comp_params)

    result = model.fit(y_fit, params, x=x_fit)

    peaks_out = []
    for i in range(len(peak_centers)):
        prefix = f'p{i}_'
        peaks_out.append({
            'center': result.params[f'{prefix}center'].value,
            'amplitude': result.params[f'{prefix}amplitude'].value,
            'sigma': result.params[f'{prefix}sigma'].value,
            'fwhm': result.params[f'{prefix}fwhm'].value,
            'height': result.params[f'{prefix}height'].value,
        })

    return {
        'peaks': peaks_out,
        'fit_result': result,
        'x_fit': x_fit,
        'y_fit': y_fit,
    }


# ---------------------------------------------------------------------------
# 6. Feature extraction — gom nhóm đỉnh gần nhau rồi fit chung
# ---------------------------------------------------------------------------
def extract_peak_features(spectrum, x, peaks, profile='voigt', overlap_threshold=15, sigma_max=30):
    """
    Từ danh sách đỉnh đã detect (output của detect_peaks) -> trả về feature
    [center, area, height, fwhm] cho mỗi đỉnh.
    """
    positions = np.sort(peaks['position'])
    if len(positions) == 0:
        return []

    groups = []
    current_group = [positions[0]]
    for p in positions[1:]:
        if p - current_group[-1] <= overlap_threshold:
            current_group.append(p)
        else:
            groups.append(current_group)
            current_group = [p]
    groups.append(current_group)

    features = []
    for group in groups:
        try:
            fit_out = fit_peak(spectrum, x, group, profile=profile, sigma_max=sigma_max)
        except Exception as e:
            print(f"[WARN] fit failed for group {group}: {e}")
            continue
        for peak in fit_out['peaks']:
            features.append({
                'center': peak['center'],
                'area': peak['amplitude'],
                'height': peak['height'],
                'fwhm': peak['fwhm'],
            })

    return features