import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import scipy.stats as stats
from sklearn.inspection import permutation_importance
import os
import logging
from datetime import datetime

# ==============================================================================
# 1. SETUP & HELPER FUNCTIONS
# ==============================================================================

# Khởi tạo Logger (Giả định logging đã được config từ bên ngoài hoặc dùng default)
logger = logging.getLogger(__name__)

def save_plot(fig, filename):
    """
    Hàm hỗ trợ lưu biểu đồ và đóng figure để giải phóng RAM.
    Lưu vào thư mục 'reports/figure' (tự động tạo nếu chưa có).
    """
    save_dir = 'reports/figures'
    os.makedirs(save_dir, exist_ok=True)
    
    path = os.path.join(save_dir, filename)
    try:
        fig.savefig(path, dpi=300, bbox_inches='tight')
        logger.info(f">> Đã lưu biểu đồ: {path}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu biểu đồ {filename}: {e}")
    finally:
        plt.close(fig) # Quan trọng: Đóng figure để tránh leak memory

# ==============================================================================
# 2. CORE EDA FUNCTIONS (GIỮ NGUYÊN KHÔNG THAY ĐỔI)
# ==============================================================================

def analyze_categorical_feature(df, col_name):
    """
    Hàm phân tích chuyên sâu cho một biến định danh (Categorical).
    (Đã tinh chỉnh để xử lý High Cardinality và dùng save_plot)
    """
    if col_name not in df.columns:
        logger.warning(f"Cảnh báo: Cột '{col_name}' không tồn tại trong DataFrame.")
        return

    # 1. Tính toán thống kê
    # Lấy top 20 giá trị phổ biến nhất để vẽ biểu đồ cho đẹp nếu dữ liệu quá nhiều
    top_n = 20
    val_counts = df[col_name].value_counts(dropna=False).reset_index()
    val_counts.columns = ['Value', 'Count']
    val_counts['Percent'] = (val_counts['Count'] / len(df)) * 100

    # 2. Ghi báo cáo thống kê vào LOG
    logger.info(f"\n{'='*10} PHÂN TÍCH CỘT: {col_name} {'='*10}")
    logger.info(f"Số lượng giá trị duy nhất (Cardinality): {df[col_name].nunique()}")

    # Nếu quá nhiều dòng, chỉ log top 20 và cảnh báo
    if len(val_counts) > top_n:
        logger.info(f"(Chỉ hiển thị Top {top_n} giá trị phổ biến nhất)")
        logger.info("-" * 40)
        logger.info(val_counts.head(top_n).to_string())
    else:
        logger.info("-" * 40)
        logger.info(val_counts.to_string())
    logger.info("-" * 40)

    # 3. Vẽ biểu đồ (Visualization)
    # Chuẩn bị dữ liệu vẽ: Chỉ lấy Top N để biểu đồ không bị rối
    plot_data = val_counts.head(top_n).copy()

    # Convert sang string để tránh lỗi sort trục x nếu dữ liệu hỗn hợp
    plot_data['Value'] = plot_data['Value'].astype(str)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Xử lý lỗi legend của seaborn (giữ nguyên ý tưởng của bạn)
    sns.barplot(data=plot_data, x='Value', y='Count', palette='viridis', hue='Value', ax=ax)
    if ax.get_legend(): ax.get_legend().remove()

    ax.set_title(f'Phân phối giá trị: {col_name} (Top {top_n})', fontsize=14)
    ax.set_xlabel(col_name)
    ax.set_ylabel('Số lượng (Tin đăng)')

    # Xoay trục x nếu nhãn quá dài
    plt.xticks(rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()

    # 4. Lưu ảnh (Dùng hàm save_plot tiện ích đã viết ở phần trước)
    save_plot(fig, f'cat_{col_name}.png')

def analyze_numerical_feature(df, column_name, unit="đơn vị"):
    """
    Phân tích thống kê mô tả và lưu biểu đồ cho biến số.
    (Đã nâng cấp để dùng logger và save_plot)
    """
    if column_name not in df.columns:
        logger.warning(f"Cảnh báo: Cột {column_name} không tồn tại trong DataFrame.")
        return

    # 1. Kiểm tra dữ liệu
    series = df[column_name].dropna()
    n_missing = df[column_name].isnull().sum()
    n_zeros = (df[column_name] == 0).sum()

    # 2. Tính toán các chỉ số thống kê
    desc = series.describe()
    skew_val = skew(series)
    kurt_val = kurtosis(series)

    # 3. Ghi log báo cáo (Thay vì print thuần)
    logger.info(f"\n{'='*10} PHÂN TÍCH CỘT: {column_name} {'='*10}")
    logger.info(f"1. Tổng quan:")
    logger.info(f"   - Số lượng bản ghi: {len(df)}")
    logger.info(f"   - Missing Values: {n_missing} ({n_missing/len(df)*100:.2f}%)")
    logger.info(f"   - Zeros (Giá trị 0): {n_zeros} ({n_zeros/len(df)*100:.2f}%)")
    logger.info(f"2. Xu hướng tập trung:")
    logger.info(f"   - Mean: {desc['mean']:.2f} {unit} | Median: {desc['50%']:.2f} {unit}")
    if not series.mode().empty:
         logger.info(f"   - Mode: {series.mode()[0]:.2f} {unit}")
    logger.info(f"3. Độ phân tán & Hình dáng:")
    logger.info(f"   - Std: {desc['std']:.2f} | Min-Max: {desc['min']:.2f} - {desc['max']:.2f} {unit}")
    logger.info(f"   - Skewness: {skew_val:.4f} ({'Lệch phải' if skew_val > 0 else 'Lệch trái'})")
    logger.info(f"   - Kurtosis: {kurt_val:.4f}")

    # 4. Vẽ và lưu biểu đồ
    fig, (ax_box, ax_hist) = plt.subplots(2, 1, sharex=True,
                                          gridspec_kw={"height_ratios": (.15, .85)},
                                          figsize=(10, 6))

    sns.boxplot(x=series, ax=ax_box, color='lightblue')
    ax_box.set(xlabel='')
    ax_box.set_title(f'Phân phối của {column_name}', fontsize=14)

    sns.histplot(series, ax=ax_hist, kde=True, color='skyblue', bins=30)
    ax_hist.axvline(desc['mean'], color='red', linestyle='--', label=f'Mean: {desc["mean"]:.2f}')
    ax_hist.axvline(desc['50%'], color='green', linestyle='-', label=f'Median: {desc["50%"]:.2f}')
    ax_hist.set_xlabel(f"{column_name} ({unit})")
    ax_hist.set_ylabel("Tần suất")
    ax_hist.legend()

    plt.tight_layout()
    # Lưu file thay vì show
    save_plot(fig, f"phan_phoi_{column_name}.png")

def check_multicollinearity(df, features_to_check):
    """
    Kiểm tra đa cộng tuyến bằng Heatmap và VIF.
    """
    logger.info("\n--- KIỂM TRA ĐA CỘNG TUYẾN (MULTICOLLINEARITY) ---")

    # Lọc các cột tồn tại và loại bỏ NaN
    valid_cols = [col for col in features_to_check if col in df.columns]
    X = df[valid_cols].dropna()

    if X.empty:
        logger.warning("Không đủ dữ liệu để tính VIF.")
        return

    # 1. Vẽ Heatmap và lưu
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = X.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, vmin=-1, vmax=1, fmt='.2f', ax=ax)
    ax.set_title('Ma trận Tương quan (Correlation Matrix)', fontsize=14)
    save_plot(fig, "heatmap_correlation.png")

    # 2. Tính VIF
    logger.info("Đang tính toán VIF...")
    X_with_const = add_constant(X)
    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_with_const.columns
    vif_data["VIF"] = [variance_inflation_factor(X_with_const.values, i)
                       for i in range(len(X_with_const.columns))]

    # Loại bỏ hằng số và sort
    vif_data = vif_data[vif_data["Feature"] != "const"].sort_values(by="VIF", ascending=False)

    logger.info("\n=== KẾT QUẢ VIF (VARIANCE INFLATION FACTOR) ===")
    logger.info("Quy tắc: VIF > 5 (Cảnh báo), VIF > 10 (Nguy hiểm - Cần loại bỏ)")
    logger.info("-" * 60)
    logger.info(vif_data.to_string(index=False))

def analyze_correlations(df, numeric_features, target_col='Log_Gia'):
    """
    Phân tích tương quan Pearson và vẽ Scatter Plot với biến mục tiêu.
    """
    logger.info(f"\n--- PHÂN TÍCH TƯƠNG QUAN VỚI {target_col} ---")

    valid_features = [f for f in numeric_features if f in df.columns]
    if target_col not in df.columns:
        logger.error(f"Biến mục tiêu {target_col} không tồn tại.")
        return

    # 1. Tính Pearson R
    # Chỉ lấy các cột số
    temp_df = df[valid_features + [target_col]].select_dtypes(include=[np.number])
    correlations = temp_df.corr()[target_col].sort_values(ascending=False).drop(target_col)

    logger.info("HỆ SỐ TƯƠNG QUAN PEARSON (r):")
    logger.info(correlations.to_string())

    # 2. Vẽ Scatter Plot (Regplot)
    # Tự động tính số dòng/cột cho grid dựa trên số lượng biến
    n_cols = len(valid_features)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))

    # Xử lý trường hợp chỉ có 1 biến (axes không phải array)
    if n_cols == 1: axes = [axes]

    for i, feature in enumerate(valid_features):
        sns.regplot(x=feature, y=target_col, data=df, ax=axes[i],
                    scatter_kws={'alpha':0.1, 's':10},
                    line_kws={'color':'red'})
        r_value = correlations.get(feature, 0)
        axes[i].set_title(f'{feature}\n(r={r_value:.2f})', fontsize=12)
        axes[i].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    save_plot(fig, f"scatter_correlation_{target_col}.png")

def analyze_categorical_anova(df, cat_features, target_col='Log_Gia'):
    """
    Phân tích ANOVA và vẽ Boxplot ngang cho các biến phân loại.
    """
    logger.info("\n=== BÁO CÁO PHÂN TÍCH ANOVA (ONE-WAY) ===")
    logger.info(f"{'Feature':<20} | {'F-Score':<10} | {'P-value':<15} | {'Kết luận'}")
    logger.info("-" * 75)

    if target_col not in df.columns:
        logger.error(f"Chưa có cột {target_col} để phân tích ANOVA.")
        return

    # Cấu hình giao diện vẽ
    sns.set_theme(style="whitegrid")

    for cat_col in cat_features:
        if cat_col not in df.columns:
            continue

        data_clean = df.dropna(subset=[cat_col, target_col])
        groups = [d for k, d in data_clean.groupby(cat_col)[target_col]]

        if len(groups) < 2:
            continue

        # Tính ANOVA
        f_score, p_value = stats.f_oneway(*groups)
        p_str = "< 0.001" if p_value < 0.001 else f"{p_value:.4f}"
        star = "⭐⭐⭐" if p_value < 0.001 else ("⭐" if p_value < 0.05 else "")

        logger.info(f"{cat_col:<20} | {f_score:<10.2f} | {p_str:<15} | {star}")

        # Vẽ Boxplot Ngang
        order = data_clean.groupby(cat_col)[target_col].median().sort_values().index

        # Dynamic Height: Logic rất hay của bạn, tôi giữ nguyên
        dynamic_height = max(5, len(order) * 0.6)
        fig, ax = plt.subplots(figsize=(14, dynamic_height))

        sns.boxplot(y=cat_col, x=target_col, data=data_clean, order=order,
                    palette='Spectral_r', hue=cat_col, legend=False,
                    linewidth=1.5, fliersize=2, orient='h', ax=ax)

        ax.set_title(f'Phân hóa giá theo: {cat_col.upper()} (F={f_score:.1f}, p={p_str})',
                     fontsize=15, fontweight='bold', color='#333333', pad=15)
        ax.set_xlabel(target_col, fontsize=12)
        ax.set_ylabel('')
        ax.grid(axis='x', linestyle='--', alpha=0.7)
        sns.despine(trim=True, offset=10, left=True)

        save_plot(fig, f"boxplot_{cat_col}_anova.png")

def verify_corner_unit_hypothesis(df):
    """
    Kiểm định giả thuyết: Missing Value trong 'CanGoc' có phải là 'Căn thường' không?
    Logic: Nếu Căn Góc (Phải) đắt hơn hẳn KXĐ -> KXĐ chính là Căn thường.
    """
    logger.info("\n--- KIỂM ĐỊNH GIẢ THUYẾT: CĂN GÓC vs. CĂN THƯỜNG (MISSING) ---")

    # Tạo bản sao để xử lý tạm thời
    df_check = df.copy()

    # Fill tạm 'KXĐ' nếu chưa fill (để code chạy an toàn)
    df_check['CanGoc'] = df_check['CanGoc'].fillna('KXĐ')

    # Lọc rác giá thấp để công bằng (sử dụng cột Gia gốc)
    if 'Gia' in df_check.columns and 'Log_Gia' in df_check.columns:
        df_check = df_check[df_check['Gia'] >= 0.8]
    else:
        logger.warning("Thiếu cột 'Gia' hoặc 'Log_Gia'. Bỏ qua kiểm định.")
        return

    # 1. Kiểm tra tỷ lệ
    counts = df_check['CanGoc'].value_counts(normalize=True) * 100
    kxd_percent = counts.get('KXĐ', 0)
    logger.info(f"Tỷ lệ phân bố: KXĐ chiếm {kxd_percent:.2f}% (Thị trường: Căn thường chiếm ~75-85%)")

    # 2. Kiểm tra chênh lệch giá (Premium Gap)
    group_phai = df_check[df_check['CanGoc'] == 'Phải']['Log_Gia']
    group_kxd = df_check[df_check['CanGoc'] == 'KXĐ']['Log_Gia']

    if group_phai.empty or group_kxd.empty:
        logger.warning("Không đủ dữ liệu hai nhóm để kiểm định T-test.")
        return

    # Tính Premium Gap %
    gap_percent = (np.expm1(group_phai.median()) - np.expm1(group_kxd.median())) / np.expm1(group_kxd.median()) * 100

    logger.info(f"Median Log(Gia) [Phải]: {group_phai.median():.2f}")
    logger.info(f"Median Log(Gia) [KXĐ] : {group_kxd.median():.2f}")
    logger.info(f"Chênh lệch giá (Premium Gap): +{gap_percent:.1f}%")

    # 3. T-Test (One-sided)
    t_stat, p_val = stats.ttest_ind(group_phai, group_kxd, equal_var=False, alternative='greater')
    logger.info(f"P-value (T-test): {p_val:.5f}")

    # Kết luận
    if p_val < 0.05 and gap_percent > 0:
        logger.info("=> KẾT LUẬN: XÁC NHẬN! 'KXĐ' chính là 'Căn thường'. (Có ý nghĩa thống kê)")
    else:
        logger.warning("=> KẾT LUẬN: NGHI NGỜ. Không thấy chênh lệch giá rõ ràng.")

    # 4. Trực quan hóa
    fig, ax = plt.subplots(figsize=(10, 6))

    # Chỉ vẽ 2 nhóm quan tâm
    plot_data = df_check[df_check['CanGoc'].isin(['KXĐ', 'Phải'])]

    sns.boxplot(x='CanGoc', y='Log_Gia', data=plot_data,
                order=['KXĐ', 'Phải'],
                palette={'KXĐ': '#95a5a6', 'Phải': '#e74c3c'},
                hue='CanGoc', legend=False, ax=ax) # Fix hue warning

    ax.set_title(f'Kiểm chứng: Căn Góc vs. KXĐ\nPremium Gap: +{gap_percent:.1f}% (p={p_val:.4f})', fontsize=14)
    ax.set_xlabel('Trạng thái (KXĐ = Missing Values)')
    ax.set_ylabel('Log(Giá)')

    save_plot(fig, "hypothesis_CanGoc.png")

def verify_direction_premium(df):
    """
    Kiểm định giả thuyết: Hướng Mát (Đông/Nam) có đắt hơn Hướng Nắng (Tây/Bắc)?
    """
    logger.info("\n--- KIỂM ĐỊNH GIẢ THUYẾT: HƯỚNG BAN CÔNG (PREMIUM CHECK) ---")

    df_check = df.copy()
    if 'HuongBanCong' not in df_check.columns:
        return

    # Debug giá trị thực tế
    unique_dirs = df_check['HuongBanCong'].unique()
    logger.info(f"Các hướng hiện có: {unique_dirs}")

    # Hàm gom nhóm (Inner function)
    def group_direction(direction):
        d = str(direction).strip()
        if d in ['Đông Nam', 'Đông', 'Nam']:
            return 'DongNam_Premium' # Nhóm Mát
        elif d in ['Tây', 'Tây Bắc', 'Bắc', 'Tây Nam']:
            return 'TayBac_Baseline' # Nhóm Nắng
        return 'Other'

    df_check['Huong_Group'] = df_check['HuongBanCong'].apply(group_direction)

    # Lấy dữ liệu 2 nhóm
    group_premium = df_check[df_check['Huong_Group'] == 'DongNam_Premium']['Log_Gia'].dropna()
    group_baseline = df_check[df_check['Huong_Group'] == 'TayBac_Baseline']['Log_Gia'].dropna()

    n_premium = len(group_premium)
    n_baseline = len(group_baseline)
    logger.info(f"Số lượng mẫu: Premium (Mát)={n_premium} | Baseline (Nắng)={n_baseline}")

    if n_premium > 5 and n_baseline > 5:
        # T-test
        t_stat, p_val = stats.ttest_ind(group_premium, group_baseline, equal_var=False, alternative='greater')

        mean_premium = group_premium.mean()
        mean_baseline = group_baseline.mean()

        logger.info(f"Mean Log(Gia) [Mát] : {mean_premium:.4f}")
        logger.info(f"Mean Log(Gia) [Nắng]: {mean_baseline:.4f}")
        logger.info(f"P-value: {p_val:.5f}")

        if p_val < 0.05:
            logger.info("=> KẾT LUẬN: BÁC BỎ H0. Hướng Mát có giá cao hơn thật sự.")
        else:
            logger.info("=> KẾT LUẬN: CHẤP NHẬN H0. Không đủ bằng chứng về chênh lệch giá.")

        # Vẽ biểu đồ (Bổ sung thêm so với code gốc để báo cáo đẹp hơn)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Huong_Group', y='Log_Gia', data=df_check[df_check['Huong_Group'] != 'Other'],
                    order=['TayBac_Baseline', 'DongNam_Premium'],
                    palette={'TayBac_Baseline': '#F39C12', 'DongNam_Premium': '#27AE60'},
                    hue='Huong_Group', legend=False, ax=ax)

        ax.set_title(f'So sánh giá: Hướng Mát (Premium) vs. Hướng Nắng (Baseline)\np-value: {p_val:.5f}', fontsize=14)
        ax.set_ylabel('Log(Giá)')
        save_plot(fig, "hypothesis_HuongBanCong.png")

    else:
        logger.warning("Không đủ mẫu cho mỗi nhóm để kiểm định.")

# ==============================================================================
# 3. EDA PIPELINE STEPS 
# ==============================================================================

def process_price_per_m2(df):
    """Xử lý Gia_m2: Tạo đơn vị triệu, lọc rác < 10tr, Log transform."""
    logger.info("\n--- BẮT ĐẦU XỬ LÝ: GIA_M2 ---")

    # Feature Engineering sơ khởi
    df['Gia_m2_trieu'] = df['Gia_m2'] * 1000
    analyze_numerical_feature(df, 'Gia_m2_trieu', unit='Triệu VNĐ')

    # Lọc rác
    initial_len = len(df)
    df = df[df['Gia_m2_trieu'] >= 10].copy()
    logger.info(f"Cleaning: Loại bỏ {initial_len - len(df)} dòng rác (< 10 triệu/m2). Còn lại: {len(df)}")

    # Log Transform
    df['Log_Gia_m2_trieu'] = np.log1p(df['Gia_m2_trieu'])

    # Vẽ so sánh
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Trước
    sns.histplot(df['Gia_m2_trieu'], kde=True, ax=axes[0], color='salmon', bins=50)
    axes[0].set_title(f"TRƯỚC: Phân phối Gốc\nSkew: {skew(df['Gia_m2_trieu']):.2f} | Kurt: {kurtosis(df['Gia_m2_trieu']):.2f}")
    axes[0].set_xlabel('Giá (Triệu/m2)')

    # Sau
    sns.histplot(df['Log_Gia_m2_trieu'], kde=True, ax=axes[1], color='skyblue', bins=50)
    axes[1].set_title(f"SAU: Log Transform\nSkew: {skew(df['Log_Gia_m2_trieu']):.2f} | Kurt: {kurtosis(df['Log_Gia_m2_trieu']):.2f}")
    axes[1].set_xlabel('Log(Giá)')

    plt.tight_layout()
    save_plot(fig, "sosanh_log_Gia_m2_trieu.png")

    return df

def process_total_price(df):
    """Xử lý Gia (Tổng): Lọc < 0.8 tỷ, Log transform, Segmentation."""
    logger.info("\n--- BẮT ĐẦU XỬ LÝ: TỔNG GIÁ (GIA) ---")

    analyze_numerical_feature(df, 'Gia', unit='Tỷ VNĐ')

    # Hard Cleaning
    initial_count = len(df)
    df = df[df['Gia'] >= 0.8].copy()
    logger.info(f"Cleaning: Loại bỏ {initial_count - len(df)} dòng (< 0.8 Tỷ). Còn lại: {len(df)}")

    # Log Transform
    df['Log_Gia'] = np.log1p(df['Gia'])
    logger.info("Transformation: Đã tạo cột Log_Gia.")

    # Segmentation
    df['Phan_Khuc'] = df['Gia'].apply(lambda x: 'Luxury_Segment' if x >= 10 else 'Mass_Market')

    # Thống kê phân khúc
    seg_stats = df['Phan_Khuc'].value_counts(normalize=True)
    logger.info(f"Segmentation Stats:\n{df['Phan_Khuc'].value_counts()}")
    if 'Luxury_Segment' in seg_stats:
        logger.info(f"Tỷ lệ Luxury Segment: {seg_stats['Luxury_Segment']*100:.2f}%")

    # Vẽ so sánh
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(df['Gia'], kde=True, ax=axes[0], color='#E76F51', bins=50)
    axes[0].set_title(f"TRƯỚC: Gốc\nSkew: {skew(df['Gia']):.2f}")
    axes[0].set_xlabel('Giá (Tỷ VNĐ)')

    sns.histplot(df['Log_Gia'], kde=True, ax=axes[1], color='#2A9D8F', bins=50)
    axes[1].set_title(f"SAU: Log Transform\nSkew: {skew(df['Log_Gia']):.2f}")
    axes[1].set_xlabel('Log(Giá)')

    plt.tight_layout()
    save_plot(fig, "sosanh_log_Tong_Gia.png")

    return df

def process_features_rooms(df):
    """Xử lý PhongNgu (Clip 4) và PhongTam (Clip 3)."""
    logger.info("\n--- BẮT ĐẦU XỬ LÝ: PHÒNG NGỦ & PHÒNG TẮM ---")

    analyze_numerical_feature(df, 'PhongNgu', unit='phòng')

    # 1. Phòng ngủ: Clip & Feature Engineering
    df['PhongNgu_Clipped'] = df['PhongNgu'].clip(upper=4)
    df['DT_per_PN'] = df['DienTich_m2'] / df['PhongNgu'] # Có thể sinh ra inf nếu PhongNgu=0, cần lưu ý

    # Xử lý trường hợp chia cho 0 nếu có (Data cleaning bổ sung an toàn)
    df.loc[np.isinf(df['DT_per_PN']), 'DT_per_PN'] = 0

    # Vẽ biểu đồ Phòng ngủ
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sns.countplot(x=df['PhongNgu_Clipped'], ax=axes[0], color='teal')
    axes[0].set_title('Phân phối Số Phòng Ngủ (Clipped 4+)')
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                          ha='center', va='bottom')

    sns.histplot(df['DT_per_PN'], kde=True, ax=axes[1], color='orange', bins=30)
    axes[1].set_title('Phân phối: Diện tích / Phòng ngủ')

    plt.tight_layout()
    save_plot(fig, "feature_PhongNgu_DT_per_PN.png")

    logger.info("Đã xử lý Phòng ngủ: Clipped max 4 & Tạo biến DT_per_PN")
    logger.info(f"Thống kê DT_per_PN:\n{df['DT_per_PN'].describe()}")

    # 2. Phòng tắm: Clip
    analyze_numerical_feature(df, 'PhongTam', unit='phòng')
    df['PhongTam_Clipped'] = df['PhongTam'].clip(upper=3)
    logger.info("Đã xử lý Phòng tắm: Clipped max 3")

    return df

def run_categorical_analysis(df):
    """Thực thi phân tích cho tất cả các cột phân loại chính."""
    logger.info("\n--- BẮT ĐẦU PHÂN TÍCH BIẾN CATEGORICAL ---")

    # Danh sách các cột cần phân tích
    categorical_columns = [
        'TinhTrangBDS', 'Loai', 'GiayTo', 'TinhTrangNoiThat',
        'HuongCuaChinh', 'HuongBanCong', 'CanGoc', 'Quan'
    ]

    # Lọc chỉ lấy các cột thực sự tồn tại trong df để tránh lỗi
    existing_cols = [col for col in categorical_columns if col in df.columns]

    if not existing_cols:
        logger.warning("Không tìm thấy cột categorical nào trong danh sách yêu cầu.")
        return df

    for col in existing_cols:
        analyze_categorical_feature(df, col)

    return df

def run_advanced_analysis(df):
    """Chạy các phân tích nâng cao: VIF, Correlation, ANOVA."""

    # 1. Kiểm tra đa cộng tuyến (Các biến X features)
    # Lưu ý: Loại bỏ biến Target và các biến bị rò rỉ (Leakage) nếu cần
    features_vif = ['DienTich_m2', 'Gia_m2', 'PhongNgu', 'PhongTam']
    check_multicollinearity(df, features_vif)

    # 2. Kiểm tra tương quan số học với Target
    # Cần đảm bảo các cột này tồn tại trong df sau các bước xử lý trước
    core_numerical = [
        'DienTich_m2', 'Gia_m2_trieu', 'Log_Gia_m2_trieu',
        'PhongNgu_Clipped', 'DT_per_PN', 'PhongTam_Clipped'
    ]
    analyze_correlations(df, core_numerical, target_col='Log_Gia')

    # 3. Phân tích ANOVA cho biến phân loại
    cat_features = [
        'Phan_Khuc', 'Quan', 'Loai', 'GiayTo',
        'TinhTrangNoiThat', 'HuongCuaChinh', 'HuongBanCong',
        'CanGoc', 'TinhTrangBDS'
    ]
    analyze_categorical_anova(df, cat_features, target_col='Log_Gia')

    return df

def run_hypothesis_testing(df):
    """Chạy các kiểm định giả thuyết kinh doanh."""
    logger.info("\n=== BẮT ĐẦU KIỂM ĐỊNH GIẢ THUYẾT (HYPOTHESIS TESTING) ===")

    # 1. Kiểm định Căn Góc (Xử lý Missing Value logic)
    verify_corner_unit_hypothesis(df)

    # 2. Kiểm định Hướng Ban Công (Feature Importance logic)
    verify_direction_premium(df)

    return df

def run_eda_pipeline(df):
    logger.info("STARTING EDA PIPELINE...")

    # Bước 1: Giá m2
    df = process_price_per_m2(df)

    # Bước 2: Tổng giá & Phân khúc
    df = process_total_price(df)

    # Bước 3: Room features
    df = process_features_rooms(df)

    # Bước 4: Categorical Analysis
    df = run_categorical_analysis(df)

    # Bước 5: Advanced Analysis (VIF, ANOVA)
    run_advanced_analysis(df)

    # Bước 6: Hypothesis Testing (MỚI THÊM)
    # Bước này giúp confirm các giả định trước khi chốt phương án Feature Engineering
    run_hypothesis_testing(df)

    logger.info("\nPIPELINE HOÀN TẤT. Vui lòng kiểm tra thư mục 'report/'.")
    return df

# ==============================================================================
# 4. MODEL ANALYSIS FUNCTIONS (PHÂN TÍCH MÔ HÌNH)
# ==============================================================================


def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    """
    Vẽ biểu đồ phân tán (Scatter Plot) so sánh Giá trị Thực tế vs Dự đoán.
    Giúp đánh giá trực quan độ chính xác của mô hình.
    """
    logger.info(f"\n--- ĐANG VẼ BIỂU ĐỒ ACTUAL vs PREDICTED ({model_name}) ---")
    
    plt.figure(figsize=(10, 8))
    # Vẽ scatter
    plt.scatter(y_true, y_pred, alpha=0.5, color='royalblue', edgecolor='w', s=40, label='Data points')
    
    # Vẽ đường chéo 45 độ (Dự đoán hoàn hảo)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction (Ideal)')
    
    plt.xlabel("Giá trị Thực tế (Actual)", fontsize=12)
    plt.ylabel("Giá trị Dự đoán (Predicted)", fontsize=12)
    plt.title(f"Actual vs Predicted - {model_name}", fontsize=15)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Lưu biểu đồ
    save_plot(plt.gcf(), f"model_actual_vs_predicted_{model_name}.png")

def plot_residuals(y_true, y_pred, model_name="Model"):
    """
    Vẽ biểu đồ Phần dư (Residuals = Actual - Predicted).
    Kiểm tra xem lỗi có phân phối chuẩn không (Normality) và có Heteroscedasticity không.
    """
    logger.info(f"\n--- ĐANG VẼ BIỂU ĐỒ RESIDUALS ({model_name}) ---")
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Histogram phân phối lỗi
    sns.histplot(residuals, kde=True, bins=30, ax=axes[0], color='teal')
    axes[0].set_title("Phân phối Phần dư (Residuals Distribution)")
    axes[0].set_xlabel("Error (Actual - Predicted)")
    axes[0].axvline(0, color='red', linestyle='--')
    
    # 2. Scatter plot: Residuals vs Predicted
    axes[1].scatter(y_pred, residuals, alpha=0.5, color='orange', edgecolor='k')
    axes[1].axhline(0, color='r', linestyle='--', lw=2)
    axes[1].set_title("Residuals vs Predicted Values (Kiểm tra phương sai đồng nhất)")
    axes[1].set_xlabel("Predicted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    save_plot(fig, f"model_residuals_{model_name}.png")

def plot_feature_importance(model, feature_names, model_name="Model"):
    """
    Vẽ biểu đồ mức độ quan trọng của các đặc trưng (Feature Importance).
    Chỉ áp dụng cho các model dạng cây (RandomForest, XGBoost, DecisionTree).
    """
    logger.info(f"\n--- ĐANG VẼ FEATURE IMPORTANCE ({model_name}) ---")
    
    # Kiểm tra model có thuộc tính feature_importances_ không
    if not hasattr(model, "feature_importances_"):
        logger.warning(f"Model {model_name} không hỗ trợ feature_importances_.")
        return

    importances = model.feature_importances_
    
    # Xử lý trường hợp feature_names không khớp
    if len(importances) != len(feature_names):
        logger.warning(f"Số lượng feature names ({len(feature_names)}) khác số lượng importances ({len(importances)}).")
        # Cắt hoặc padding nếu cần thiết, ở đây ta lấy độ dài tối thiểu
        min_len = min(len(importances), len(feature_names))
        importances = importances[:min_len]
        feature_names = feature_names[:min_len]

    # Tạo DataFrame để sort dễ dàng
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    # Lấy Top 20 features
    top_n = 20
    df_imp = df_imp.head(top_n)

    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='viridis')
    plt.title(f"Top {top_n} Feature Importance - {model_name}", fontsize=15)
    plt.xlabel("Mức độ quan trọng (Importance Score)")
    plt.ylabel("Đặc trưng (Feature)")
    plt.tight_layout()
    
    save_plot(plt.gcf(), f"model_feature_importance_{model_name}.png")

def plot_permutation_importance(model, X_test, y_test, feature_names, model_name="Model"):
    """
    Vẽ Feature Importance cho MỌI LOẠI MODEL (SVR, Voting, Stacking...)
    Dựa trên kỹ thuật Permutation Importance.
    """
    logger.info(f"\n--- ĐANG TÍNH PERMUTATION IMPORTANCE CHO {model_name} (Sẽ hơi lâu xíu...) ---")
    
    # Tính toán (lặp lại 10 lần xáo trộn để lấy trung bình cho chuẩn)
    # n_jobs=-1 để chạy đa luồng cho nhanh
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1, scoring='neg_root_mean_squared_error'
    )
    
    # Lấy giá trị quan trọng trung bình
    importances = result.importances_mean
    
    # Xử lý nếu feature_names không khớp độ dài
    if len(importances) != len(feature_names):
         min_len = min(len(importances), len(feature_names))
         importances = importances[:min_len]
         feature_names = feature_names[:min_len]

    # Tạo DataFrame
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Lấy Top 20
    top_n = 20
    df_imp = df_imp.head(top_n)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=df_imp, palette='magma')
    plt.title(f"Permutation Importance (Top {top_n}) - {model_name}", fontsize=15)
    plt.xlabel("Mức độ ảnh hưởng đến sai số (RMSE Impact)")
    plt.ylabel("Đặc trưng (Feature)")
    plt.tight_layout()
    
    save_plot(plt.gcf(), f"model_permutation_importance_{model_name}.png")

def run_model_analysis(model, X_test, y_test, feature_names=None, model_name="Model"):
    """
    Pipeline chạy toàn bộ phần phân tích mô hình sau khi train xong.
    """
    logger.info("=== BẮT ĐẦU PHÂN TÍCH MÔ HÌNH (MODEL ANALYSIS) ===")
    
    # 1. Dự đoán
    y_pred = model.predict(X_test)
    
    # 2. Vẽ Actual vs Predicted
    plot_actual_vs_predicted(y_test, y_pred, model_name)
    
    # 3. Vẽ Residuals
    plot_residuals(y_test, y_pred, model_name)
    
    # 4. Vẽ Feature Importance (nếu có feature_names)
    if feature_names is not None:
        plot_feature_importance(model, feature_names, model_name)
        
    logger.info("=== HOÀN TẤT PHÂN TÍCH MÔ HÌNH ===")

if __name__ == "__main__":
    # Ví dụ cách chạy (nếu chạy độc lập)
    # logging.basicConfig(level=logging.INFO)
    # df = pd.read_csv('cleaned_data.csv')
    # df = run_eda_pipeline(df)
    pass