# main.py

import os
import sys
import pandas as pd
import numpy as np
import argparse
import logging

# --- IMPORT TỪ SRC (LOCAL STRUCTURE) ---
from src.utils import load_config, set_seed, setup_logger
from src.preprocessor import (
    DataLoader,
    RealEstateFeatureExtractor,
    MissingValueImputer,
    OutlierHandler,
    DataScaler,
    CategoricalEncoder
)
from src.model_trainer import ModelTrainer
# [NEW] Import module Visualization
from src.visualization import run_eda_pipeline, run_model_analysis

def main():
    # =========================================================================
    # BƯỚC 0: KHỞI TẠO MÔI TRƯỜNG & CONFIG
    # =========================================================================
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Lỗi đọc config: {e}")
        return

    # Setup Logger & Seed
    # [QUAN TRỌNG] Config logging root để visualization.py cũng ghi vào file này
    log_path = config['paths']['log_path']
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    # Thiết lập logger chính cho Main
    logger = setup_logger(log_path, name="MainPipeline")
    
    # Thiết lập cấu hình chung cho logging library để các module con (như visualization)
    # tự động ghi vào cùng file này
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        force=True # Ghi đè các config cũ nếu có
    )

    set_seed(config['data']['random_seed'])
    
    logger.info("=== BẮT ĐẦU QUY TRÌNH XỬ LÝ DỮ LIỆU (OOP PIPELINE) ===")

    # --- CẤU HÌNH ĐƯỜNG DẪN ---
    INPUT_FILE = config['paths']['raw_data']
    OUTPUT_FILE_PRE_MODEL = config['paths']['eda_path']      # File cho EDA (Visualization)
    OUTPUT_FILE_FINAL = config['paths']['processed_path']    # File cho Model

    # =========================================================================
    # BƯỚC 1: TẢI DỮ LIỆU
    # =========================================================================
    df = DataLoader.load_data(INPUT_FILE, logger=logger)

    if df.empty:
        logger.error("Dữ liệu đầu vào rỗng. Dừng pipeline.")
        return

    # Khởi tạo bộ xử lý đặc trưng BĐS
    re_extractor = RealEstateFeatureExtractor(
        district_mapping=config['preprocessing']['district_mapping'],
        unit_factors=config['preprocessing']['unit_factors'],
        logger=logger
    )
    re_extractor.process(df)

    # =========================================================================
    # BƯỚC 2: STRUCTURE CLEANING
    # =========================================================================
    logger.info(">> [Giai đoạn 1] Làm sạch cấu trúc & Xóa cột rác")
    re_extractor.drop_columns(['MaCanHo', 'TenPhanKhu'])
    re_extractor.rename_columns({'Phongngu': 'PhongNgu'})

    # =========================================================================
    # BƯỚC 3: TRÍCH XUẤT ĐẶC TRƯNG (FEATURE EXTRACTION)
    # =========================================================================
    logger.info(">> [Giai đoạn 2] Trích xuất thông tin từ dữ liệu thô")
    re_extractor.process(
        re_extractor.get_data(),
        col_addr='DiaChi',
        col_price='Gia',
        col_area='DienTich',
        target_currency='tỷ'
    )

    re_extractor.rename_columns({
        'Quan_Huyen': 'Quan',
        'Gia_Clean': 'Gia_Ty',
        'DienTich_Clean': 'DienTich_m2'
    })
    re_extractor.drop_columns(['DiaChi'])

    re_extractor.extract_room_feature('PhongNgu')
    re_extractor.extract_room_feature('PhongTam')

    re_extractor.standardize_binary(col='DacDiem', true_label='Phải', false_label='KXĐ')
    re_extractor.rename_columns({'DacDiem': 'CanGoc'})

    # =========================================================================
    # BƯỚC 4: XỬ LÝ DỮ LIỆU THIẾU
    # =========================================================================
    logger.info(">> [Giai đoạn 3] Xử lý dữ liệu thiếu")
    current_df = re_extractor.get_data()

    imputer_mode = MissingValueImputer(strategy_map={
        'TinhTrangBDS': 'mode', 'Loai': 'mode',
        'PhongNgu': 'mode', 'PhongTam': 'mode'
    }, logger=logger)
    current_df = imputer_mode.process(current_df)

    imp_noithat = MissingValueImputer({'TinhTrangNoiThat': 'constant'}, fill_value='Bàn giao thô', logger=logger)
    current_df = imp_noithat.process(current_df)

    imp_giayto = MissingValueImputer({'GiayTo': 'constant'}, fill_value=' Giấy tờ khác', logger=logger)
    current_df = imp_giayto.process(current_df)

    imp_huong = MissingValueImputer({'HuongCuaChinh': 'constant', 'HuongBanCong': 'constant'}, fill_value='Khác', logger=logger)
    current_df = imp_huong.process(current_df)

    # =========================================================================
    # BƯỚC 5: CHUẨN HÓA CATEGORY
    # =========================================================================
    logger.info(">> [Giai đoạn 4] Chuẩn hóa dữ liệu phân loại")
    re_extractor._data = current_df 

    noithat_map = {
        'Nội thất cao cấp': 'Cao cấp', 'Nội thất đầy đủ': 'Đầy đủ',
        'Hoàn thiện cơ bản': 'Cơ bản', 'Bàn giao thô': 'Thô'
    }
    re_extractor.custom_text_mapping('TinhTrangNoiThat', noithat_map)
    re_extractor.standardize_category(col='Loai', new_col='Loai')

    # =========================================================================
    # BƯỚC 6: LOGIC ĐẶC THÙ & KIỂM TRA CHÉO
    # =========================================================================
    logger.info(">> [Giai đoạn 5] Logic nghiệp vụ & Kiểm tra chéo")
    current_df = re_extractor.get_data()

    current_df['DonGia_m2'] = current_df['Gia_Ty'] / current_df['DienTich_m2']
    current_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    re_extractor._data = current_df
    re_extractor.cross_check_and_fix_area(
        area_col='DienTich_m2', price_col='Gia_Ty',
        unit_price_col='DonGia_m2', threshold=10.0
    )
    current_df = re_extractor.get_data()

    logger.info("Lọc diện tích ngoại lai (30 - 110 m2)...")
    mask_area = (current_df['DienTich_m2'] >= 30) & (current_df['DienTich_m2'] <= 110)
    current_df = current_df[mask_area]

    # =========================================================================
    # BƯỚC 7: LỌC NGOẠI LAI THỐNG KÊ
    # =========================================================================
    logger.info(">> [Giai đoạn 6] Lọc ngoại lai thống kê (IQR)")
    outlier_handler = OutlierHandler(
        columns=['DonGia_m2'], method='iqr', action='remove', logger=logger
    )
    current_df = outlier_handler.process(current_df)

    # Dọn dẹp sơ bộ để chuẩn bị lưu file EDA
    cols_garbage = ['Gia', 'Gia/m2', 'DienTich', 'SoTang'] 
    existing_garbage = [c for c in cols_garbage if c in current_df.columns]
    if existing_garbage:
        current_df.drop(columns=existing_garbage, inplace=True)

    current_df.rename(columns={
        'Gia_Ty': 'Gia',
        'DienTich_Clean': 'DienTich_m2',
        'DonGia_m2': 'Gia_m2'
    }, inplace=True)

    current_df.dropna(subset=['Gia', 'DienTich_m2'], inplace=True)
    current_df.reset_index(drop=True, inplace=True)

    # Feature Engineering
    logger.info(">> [Feature Engineering] Tạo biến tương tác")
    current_df['TongPhong'] = current_df['PhongNgu'] + current_df['PhongTam']
    current_df['DienTich_per_Phong'] = current_df['DienTich_m2'] / (current_df['PhongNgu'] + 1)
    current_df['TyLe_Ngu_Tam'] = current_df['PhongNgu'] / (current_df['PhongTam'] + 1)

    # Log Transform
    logger.info(">> [Optimization] Áp dụng Log Transform cho cột Target")
    current_df['Log_Gia'] = np.log1p(current_df['Gia'])

    # =========================================================================
    # TÁCH DỮ LIỆU CHO EDA TRƯỚC KHI XÓA CỘT LEAKAGE
    # =========================================================================
    # Copy ra một bản riêng cho EDA (vẫn còn giữ cột Gia_m2 để phân tích)
    df_eda = current_df.copy()

    # --- LƯU FILE SẠCH CHO EDA (VISUALIZATION) ---
    # Lưu bản df_eda (có đầy đủ cột) để sau này mở lên xem dễ hơn
    logger.info(f">> [SAVE] Lưu file dữ liệu sạch cho EDA: {OUTPUT_FILE_PRE_MODEL}")
    DataLoader.save_data(df_eda, OUTPUT_FILE_PRE_MODEL, logger=logger)

    # =========================================================================
    # [QUAN TRỌNG] XỬ LÝ CHO MODEL (CHẶN DATA LEAKAGE)
    # =========================================================================
    # Giờ mới xóa Gia_m2 trong current_df (bản dùng để train model)
    if 'Gia_m2' in current_df.columns:
        current_df.drop(columns=['Gia_m2'], inplace=True)
        logger.info(">> Đã xóa cột 'Gia_m2' khỏi tập dữ liệu Train để chống Leakage.")

    # =========================================================================
    # BƯỚC 7.5: CHẠY TRỰC QUAN HÓA (EDA)
    # =========================================================================
    logger.info("\n=============================================")
    logger.info("   BẮT ĐẦU VISUALIZATION (EDA PHASE)")
    logger.info("=============================================")
    try:
        # Chạy pipeline vẽ biểu đồ trên bản df_eda (vẫn còn Gia_m2)
        run_eda_pipeline(df_eda)
        logger.info(">> EDA hoàn tất. Các biểu đồ đã được lưu.")
    except Exception as e:
        logger.error(f"Lỗi trong quá trình EDA Visualization: {e}")

    # =========================================================================
    # BƯỚC 8: MÃ HÓA (ENCODING) - SCALING DỜI RA SAU
    # =========================================================================
    logger.info(">> [Giai đoạn 7] Chuẩn bị dữ liệu cho Model")
    
    # [FIX QUAN TRỌNG] Xử lý cột Target: Gán Log_Gia vào Gia rồi xóa Log_Gia
    if 'Log_Gia' in current_df.columns:
        logger.info(">> Cập nhật Target: Sử dụng Log_Gia thay cho Gia gốc.")
        # 1. Gán giá trị Log vào cột Target chính ('Gia')
        current_df['Gia'] = current_df['Log_Gia']
        
        # 2. XÓA CỘT Log_Gia ĐỂ CHẶN LEAKAGE
        current_df.drop(columns=['Log_Gia'], inplace=True) 

    # 1. Clean string (Giữ nguyên)
    current_df.columns = current_df.columns.str.strip()
    obj_cols = current_df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        current_df[c] = current_df[c].astype(str).str.strip()

    # [CHANGE] KHÔNG SCALE Ở ĐÂY NỮA ĐỂ TRÁNH LEAKAGE
    # Chúng ta chỉ định nghĩa danh sách cột cần scale để gửi cho Trainer
    cols_to_scale = ['DienTich_m2', 'PhongNgu', 'PhongTam', 
                     'TongPhong', 'DienTich_per_Phong', 'TyLe_Ngu_Tam']
    
    # logger.info("Skipping scaling in main pipeline to prevent data leakage...")

    # 2. ENCODING (Giữ nguyên - OneHot nên làm trước để đồng bộ cột)
    cols_onehot = ['Quan', 'Loai', 'GiayTo', 'CanGoc']
    existing_onehot = [c for c in cols_onehot if c in current_df.columns]
    if existing_onehot:
        encoder_oh = CategoricalEncoder(columns=existing_onehot, method='onehot', logger=logger)
        current_df = encoder_oh.process(current_df)

    cols_label = ['TinhTrangBDS', 'TinhTrangNoiThat', 'HuongCuaChinh', 'HuongBanCong']
    existing_label = [c for c in cols_label if c in current_df.columns]
    if existing_label:
        encoder_lbl = CategoricalEncoder(columns=existing_label, method='label', logger=logger)
        current_df = encoder_lbl.process(current_df)

    # =========================================================================
    # BƯỚC 9: LƯU TRỮ DỮ LIỆU (Unscaled Data)
    # =========================================================================
    logger.info(f">> [SAVE] Lưu file dữ liệu (Unscaled) cho Model: {OUTPUT_FILE_FINAL}")
    DataLoader.save_data(current_df, OUTPUT_FILE_FINAL, logger=logger)

    # =========================================================================
    # BƯỚC 10: HUẤN LUYỆN MÔ HÌNH (MODEL TRAINING)
    # =========================================================================
    logger.info("\n=== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN MODEL ===")
    
    trainer = None
    try:
        # [CHANGE] Truyền thêm cols_to_scale vào ModelTrainer
        trainer = ModelTrainer(config=config, logger=logger, cols_to_scale=cols_to_scale)
        
        # 1. Load dữ liệu
        trainer.load_data()
        
        # 2. Chia tập Train/Test
        trainer.split_data()
        
        # 3. [NEW] Scale dữ liệu SAU KHI SPLIT (Fit Train -> Transform All)
        trainer.scale_data() 
        
        # 4. Chạy Training Loop
        trainer.train_model()
        
        logger.info(f"Mô hình tốt nhất: {trainer.best_model_name}")
        
    except Exception as e:
        logger.exception(f"Lỗi xảy ra trong quá trình Training: {e}")

    # =========================================================================
    # BƯỚC 11: PHÂN TÍCH KẾT QUẢ MODEL (MODEL VISUALIZATION)
    # =========================================================================
    if trainer and trainer.best_models:
        logger.info("\n=============================================")
        logger.info("   BẮT ĐẦU PHÂN TÍCH KẾT QUẢ MÔ HÌNH")
        logger.info("=============================================")
        try:
            # 1. Lấy danh sách tên đặc trưng
            feature_names = trainer.get_feature_names()
            
            # 2. Duyệt qua TẤT CẢ các model đã train
            for model_name, model in trainer.best_models.items():
                logger.info(f">> Đang phân tích model: {model_name}")
                
                # 1. Vẽ Actual vs Predicted & Residuals (Cơ bản)
                run_model_analysis(
                    model=model,
                    X_test=trainer.X_test,
                    y_test=trainer.y_test,
                    feature_names=feature_names,
                    model_name=model_name
                )
                
                # 2. Vẽ Feature Importance (Nâng cao)
                # Nếu model có sẵn feature_importances_ (RF, XGBoost) -> Dùng cái có sẵn cho nhanh
                if hasattr(model, "feature_importances_"):
                    from src.visualization import plot_feature_importance
                    plot_feature_importance(model, feature_names, model_name)
                
                # Nếu model KHÔNG có sẵn (Voting, SVR) -> Dùng Permutation Importance
                else:
                    logger.info(f"Model {model_name} không có feature_importances_. Chuyển sang dùng Permutation Importance.")
                    from src.visualization import plot_permutation_importance
                    
                    # Lưu ý: Cần truyền X_test và y_test để nó tính toán
                    plot_permutation_importance(
                        model=model, 
                        X_test=trainer.X_test, 
                        y_test=trainer.y_test, 
                        feature_names=feature_names, 
                        model_name=model_name
                    )

            logger.info(">> Model Analysis hoàn tất.")
                
        except Exception as e:
            logger.error(f"Lỗi trong quá trình Model Visualization: {e}", exc_info=True)
    
    logger.info("=== PIPELINE KẾT THÚC THÀNH CÔNG ===") 

if __name__ == "__main__":
    main()