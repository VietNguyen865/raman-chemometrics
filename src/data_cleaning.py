"""
data_cleaning.py

KHÁC VỚI raman_processing.py:
- raman_processing.py xử lý 1 phổ đơn lẻ (thuật toán tín hiệu, tái dùng
  được cho bất kỳ dataset Raman nào).
- File này gắn chặt với QUY ƯỚC ĐẶT TÊN và QUYẾT ĐỊNH THÍ NGHIỆM cụ thể
  của Ethanol_Methanol.xlsx: parse tên cột thành nồng độ, chạy SNR cho
  toàn bộ dataset, và quyết định mẫu nào nên loại khỏi phân tích.

Quy ước tên cột trong Ethanol_Methanol.xlsx:
    'Ethanol'   -> dãy E, nồng độ ethanol 100%
    'Methanol'  -> dãy E, nồng độ ethanol 0%
    'E90_a'     -> dãy E (pha loãng ethanol/methanol), nồng độ 90%, lần đo 'a'
    'E5_a'      -> dãy E, nồng độ 5%, lần đo 'a'
    'EM8_c'     -> dãy EM (hỗn hợp ethanol-methanol theo mã số 1-9), mã 8,
                   lần đo 'c'
    'EMX', 'EMX_b', 'EMX_c' -> dãy EM, mức đặc biệt 'X' (không rõ tỉ lệ số,
                   giữ nguyên nhãn 'X' thay vì ép về số)
"""
import re
import numpy as np
import pandas as pd

from raman_processing import baseline_correction, calc_snr_std


# ---------------------------------------------------------------------------
# 1. Parse tên cột -> nhãn thí nghiệm
# ---------------------------------------------------------------------------
_LABEL_PATTERN = re.compile(r"^(?P<series>EM|E)(?P<level>\d+|X)?(?:_(?P<replicate>[a-c]))?$")


def parse_sample_label(col_name):
    """
    Parse tên cột thành dict {series, concentration, replicate, raw}.

    concentration: số (float) nếu parse được, None nếu là mức 'X' đặc biệt.
    replicate: 'a'/'b'/'c', hoặc None nếu không có hậu tố lặp (Ethanol,
               Methanol, EMX gốc).

    Ví dụ:
        'E90_a'  -> {'series': 'E',  'concentration': 90.0, 'replicate': 'a'}
        'Ethanol'-> {'series': 'E',  'concentration': 100.0,'replicate': None}
        'Methanol'->{'series': 'E',  'concentration': 0.0,  'replicate': None}
        'EM8_c'  -> {'series': 'EM', 'concentration': 8.0,  'replicate': 'c'}
        'EMX_b'  -> {'series': 'EM', 'concentration': None, 'replicate': 'b'}
    """
    if col_name == "Ethanol":
        return {"series": "E", "concentration": 100.0, "replicate": None, "raw": col_name}
    if col_name == "Methanol":
        return {"series": "E", "concentration": 0.0, "replicate": None, "raw": col_name}

    m = _LABEL_PATTERN.match(col_name)
    if not m:
        raise ValueError(f"Không parse được tên cột: {col_name!r}")

    series = m.group("series")
    level = m.group("level")
    replicate = m.group("replicate")

    if level is None or level == "X":
        concentration = None
    else:
        concentration = float(level)

    return {"series": series, "concentration": concentration, "replicate": replicate, "raw": col_name}


def build_label_table(sample_cols):
    """Parse toàn bộ danh sách tên cột -> DataFrame nhãn, tiện lọc/group."""
    rows = [parse_sample_label(c) for c in sample_cols]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Tính SNR cho toàn bộ dataset -> báo cáo để audit
# ---------------------------------------------------------------------------
# 4 đỉnh mạnh, đã xác nhận thực nghiệm là đỉnh thật (không phải nhiễu) trên
# phổ Ethanol nguyên chất: dùng làm điểm tham chiếu SNR chung cho mọi mẫu.
REFERENCE_PEAKS = [660.3, 1094.8, 1642.8, 3075.3]


def build_snr_report(df, x, sample_cols=None, peaks=None, bg_range=(1900, 2700),
                      baseline_method="airpls", lam=1e5):
    """
    Chạy baseline_correction + calc_snr_std cho từng mẫu trong dataset,
    trả về DataFrame để audit chất lượng dữ liệu trước khi đưa vào
    augmentation/modeling.

    df         : DataFrame gốc (đã lọc NaN theo x, xem ví dụ dưới).
    x          : trục Raman shift (cm-1), cùng độ dài với các cột trong df.
    sample_cols: danh sách cột cần tính, mặc định = mọi cột trừ trục x.
    peaks      : danh sách đỉnh tham chiếu để tính SNR, mặc định REFERENCE_PEAKS.

    Trả về DataFrame gồm: raw, series, concentration, replicate,
    snr_mean (trung bình SNR qua các đỉnh tham chiếu), snr_min, snr_<peak>...
    """
    if sample_cols is None:
        sample_cols = [c for c in df.columns if c != "Raman Shift (cm-1)"]
    if peaks is None:
        peaks = REFERENCE_PEAKS

    x = np.asarray(x, dtype=float)
    rows = []

    for col in sample_cols:
        y = df[col].values.astype(float)
        corrected = baseline_correction(y, method=baseline_method, lam=lam)

        label = parse_sample_label(col)
        snr_values = {}
        for p in peaks:
            snr_values[f"snr_{p}"] = calc_snr_std(corrected, x, p, bg_range=bg_range)

        row = dict(label)
        row.update(snr_values)
        row["snr_mean"] = float(np.nanmean(list(snr_values.values())))
        row["snr_min"] = float(np.nanmin(list(snr_values.values())))
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Gắn cờ mẫu SNR bất thường (so với các lần lặp cùng nồng độ)
# ---------------------------------------------------------------------------
def flag_low_snr_samples(snr_report, ratio_threshold=0.3, min_group_size=2):
    """
    Gắn cờ mẫu có SNR thấp bất thường SO VỚI CÁC LẦN LẶP CÙNG NỒNG ĐỘ
    (không dùng ngưỡng tuyệt đối, vì SNR vốn khác nhau rất nhiều giữa các
    mức nồng độ -- ethanol nguyên chất SNR~150-190, E5_a chỉ ~13).

    Logic: với mỗi nhóm (series, concentration) có >= min_group_size lần
    lặp, tính median snr_mean của nhóm. Mẫu nào có snr_mean <
    ratio_threshold * median_nhóm thì bị gắn cờ 'flagged=True'.

    Đây chính là cách phát hiện case EM8_c (snr_mean ~11) so với
    EM8_a/EM8_b (~145-185) đã tìm ra thủ công trước đó -- giờ tự động hóa.

    ratio_threshold=0.3: mẫu thấp hơn 30% so với median nhóm bị coi là bất
    thường. Đây là ngưỡng thực nghiệm ban đầu, NÊN xem lại report bằng mắt
    trước khi áp dụng loại mẫu hàng loạt cho paper.

    Trả về snr_report kèm 2 cột mới: 'group_median_snr', 'flagged'.
    """
    report = snr_report.copy()
    report["group_median_snr"] = np.nan
    report["flagged"] = False

    group_cols = ["series", "concentration"]
    for _, group_idx in report.groupby(group_cols).groups.items():
        group = report.loc[group_idx]
        if len(group) < min_group_size:
            # không đủ lần lặp để so sánh -> không gắn cờ, chỉ ghi chú
            continue
        median_snr = group["snr_mean"].median()
        report.loc[group_idx, "group_median_snr"] = median_snr
        report.loc[group_idx, "flagged"] = group["snr_mean"] < (ratio_threshold * median_snr)

    return report


def summarize_flags(flagged_report):
    """In tóm tắt các mẫu bị gắn cờ, để review nhanh trước khi quyết định loại."""
    bad = flagged_report[flagged_report["flagged"]]
    if bad.empty:
        print("Không có mẫu nào bị gắn cờ SNR bất thường.")
        return bad
    print(f"{len(bad)} mẫu bị gắn cờ SNR bất thường so với các lần lặp cùng nồng độ:")
    for _, row in bad.iterrows():
        print(f"  {row['raw']:10s} snr_mean={row['snr_mean']:7.1f}  "
              f"(median nhóm={row['group_median_snr']:7.1f}, "
              f"tỉ lệ={row['snr_mean']/row['group_median_snr']:.2f})")
    return bad
