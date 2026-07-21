"""
diagnose_model_issues.py

Script chẩn đoán vì sao R² âm hàng loạt ở 06_model_comparison.ipynb.
Chạy trong thư mục có src/ trên PYTHONPATH (giống notebook), và
data/raw/Ethanol_Methanol.xlsx tồn tại.

3 phần:
  1. Kiểm tra split train/test (E-series) -- xem test set có bao nhiêu mẫu,
     nồng độ nào rơi vào test.
  2. Kiểm tra lệch scale cường độ giữa các lần lặp (_a vs _c) -- nghi vấn
     batch effect (laser power / thời gian tích phân khác nhau giữa các
     lần đo) làm nhiễu tín hiệu nồng độ.
  3. Leave-One-Out CV trên 20 mẫu E-series GỐC (không augment, không phụ
     thuộc 1 lần split) để cô lập vấn đề "dữ liệu quá ít/nhiễu" khỏi vấn đề
     "model/kiến trúc sai" -- so sánh raw intensity vs 3 kiểu chuẩn hoá.
"""
import sys, os, re
sys.path.insert(0, '../src')  # chỉnh nếu vị trí src/ khác

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error

from raman_processing import baseline_correction
from data_cleaning import build_label_table, parse_sample_label

RAW_XLSX = '../data/raw/Ethanol_Methanol.xlsx'
AIRPLS_LAM = 1e5  # thay bằng giá trị thật trong chosen_params.json nếu khác

# ---------------------------------------------------------------------------
# 0. Load + baseline-correct (giống bước 02)
# ---------------------------------------------------------------------------
df_raw = pd.read_excel(RAW_XLSX)
x_full = df_raw['Raman Shift (cm-1)'].values
mask = ~np.isnan(x_full)
x = x_full[mask]
df = df_raw.loc[mask].reset_index(drop=True)
sample_cols = [c for c in df.columns if c != 'Raman Shift (cm-1)']

df_clean = pd.DataFrame({
    col: baseline_correction(df[col].values.astype(float), method='airpls', lam=AIRPLS_LAM)
    for col in sample_cols
})
labels_all = build_label_table(sample_cols)

# ---------------------------------------------------------------------------
# 1. Kiểm tra split train/test cho E-series
# ---------------------------------------------------------------------------
print('=' * 70)
print('1. KIEM TRA SPLIT TRAIN/TEST (E-series)')
print('=' * 70)

has_conc = labels_all['concentration'].notna()
df_with_conc = labels_all[has_conc].copy()
df_no_conc = labels_all[~has_conc].copy()

e_series = df_with_conc[df_with_conc['series'] == 'E']
print('\nSo luong mau theo nong do (E-series):')
print(e_series.groupby('concentration').size())
print('\n-> Cac nong do chi co 1 mau (Ethanol, Methanol, E5, E10) LUON roi')
print('   vao train khi stratify -- test set chi con lay tu cac nong do co')
print('   >=2 mau, nen pham vi test se hep hon train (khong phai loi, chi')
print('   la dac diem cua du lieu it lap lai).')

def build_strat_key(df, bin_width, use_series):
    conc = df['concentration'].clip(upper=99)
    conc_bin = (conc // bin_width) * bin_width
    if use_series:
        return df['series'].astype(str) + '_' + conc_bin.astype(str)
    return conc_bin.astype(str)

candidate_configs = [(10, True), (20, True), (25, True), (20, False),
                      (25, False), (50, False), (101, False)]
strat_key = None
for bin_width, use_series in candidate_configs:
    candidate = build_strat_key(df_with_conc, bin_width, use_series)
    if candidate.value_counts().min() >= 2:
        strat_key = candidate
        break

train_raw, test_raw = train_test_split(
    df_with_conc['raw'], test_size=0.2, random_state=42, stratify=strat_key)
train_raw = pd.concat([train_raw, df_no_conc['raw']], ignore_index=True)

e_split = []
for raw, split in list(zip(train_raw, ['train'] * len(train_raw))) + \
                   list(zip(test_raw, ['test'] * len(test_raw))):
    lbl = parse_sample_label(raw)
    if lbl['series'] == 'E' and lbl['concentration'] is not None:
        e_split.append({'raw': raw, 'split': split, 'concentration': lbl['concentration']})
print('\n' + pd.DataFrame(e_split).sort_values('concentration').to_string(index=False))
print(f"\n-> Test set E-series chi co "
      f"{sum(1 for r in e_split if r['split']=='test')} mau. Voi n nho nhu vay,"
      f" R2 dao dong RAT manh chi vi 1-2 diem lech.")

# ---------------------------------------------------------------------------
# 2. Kiem tra lech scale cuong do giua cac lan lap (_a vs _c)
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('2. KIEM TRA LECH SCALE CUONG DO GIUA CAC LAN LAP (batch effect)')
print('=' * 70)

rows = []
for c in sample_cols:
    lbl = parse_sample_label(c)
    if lbl['series'] == 'E' and lbl['concentration'] is not None:
        y = df_clean[c].values
        rows.append({'col': c, 'concentration': lbl['concentration'],
                      'replicate': lbl['replicate'] or 'single',
                      'max_intensity': y.max()})
rep_df = pd.DataFrame(rows).sort_values(['replicate', 'concentration'])
print(rep_df.to_string(index=False))

print('\nTuong quan (cuong do, nong do) theo tung nhom lan lap:')
for rep, g in rep_df.groupby('replicate'):
    if len(g) >= 3:
        corr = np.corrcoef(g['concentration'], g['max_intensity'])[0, 1]
        print(f'  replicate={rep}: n={len(g)}, corr={corr:.3f}, '
              f'khoang cuong do=[{g.max_intensity.min():.0f}, {g.max_intensity.max():.0f}]')
print('\n-> Neu cac nhom lan lap co khoang cuong do lech nhau nhieu lan (vd')
print('   10x) o CUNG mot nong do, day la dau hieu ro rang cua batch effect')
print('   (laser power/thoi gian tich phan khac nhau giua cac lan do), khong')
print('   phai do nong do. Model hoc tren cuong do tuyet doi se bi nham lan.')

# ---------------------------------------------------------------------------
# 3. Leave-One-Out CV tren 20 mau E-series GOC, so sanh cac kieu chuan hoa
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('3. LEAVE-ONE-OUT CV (20 mau goc, khong augment) -- co lap van de')
print('   du lieu khoi van de model/kien truc')
print('=' * 70)

e_cols = [c for c in sample_cols
          if parse_sample_label(c)['series'] == 'E'
          and parse_sample_label(c)['concentration'] is not None]
concs = [parse_sample_label(c)['concentration'] for c in e_cols]
X_raw = df_clean[e_cols].values.T.astype(float)
y = np.array(concs)

def loo_eval(X, y, label):
    loo = LeaveOneOut()
    preds = np.zeros_like(y)
    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr_idx])
        Xte = scaler.transform(X[te_idx])
        ncomp = min(8, len(tr_idx) - 1)
        pca = PCA(n_components=ncomp, random_state=42)
        Xtr_p = pca.fit_transform(Xtr)
        Xte_p = pca.transform(Xte)
        br = BayesianRidge()
        br.fit(Xtr_p, y[tr_idx])
        preds[te_idx] = br.predict(Xte_p)
    r2 = r2_score(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    print(f'{label:22s}: LOO R2={r2:6.3f}  RMSE={rmse:6.2f}')
    return preds

loo_eval(X_raw, y, 'Raw intensity')
loo_eval(X_raw / (np.linalg.norm(X_raw, axis=1, keepdims=True) + 1e-9), y, 'L2-normalized')
loo_eval(X_raw / (X_raw.max(axis=1, keepdims=True) + 1e-9), y, 'Max-normalized')
loo_eval(X_raw / (X_raw.sum(axis=1, keepdims=True) + 1e-9), y, 'Area-normalized')

print('\n-> Neu R2 van am/gan 0 ngay ca voi LOO-CV (dung het 19/20 mau de train,')
print('   danh gia tren tung mau con lai), day la bang chung ro rang: van de')
print('   nam o LUONG/CHAT LUONG DU LIEU GOC (qua it mau, nhieu do giua cac')
print('   lan do), khong phai do chon sai kien truc CNN/Transformer/GP.')
