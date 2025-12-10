# src/preprocessor.py
import pandas as pd
import numpy as np
import re
import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.ensemble import IsolationForest
import logging

# =============================================================================
# 1. ABSTRACT BASE CLASS
# =============================================================================
class DataProcessor(ABC):
    """
    Interface chuẩn cho mọi processor trong pipeline xử lý dữ liệu.
    Nhiệm vụ chính: 
    - Quản lý trạng thái dữ liệu nội bộ (self._data).
    - Tích hợp logger tập trung từ bên ngoài (Dependency Injection).
    """
    def __init__(self, logger: Optional[logging.Logger] = None):
        self._data: Optional[pd.DataFrame] = None
        # Nếu không truyền logger vào, tạo một logger mặc định để tránh lỗi NoneType
        self.logger = logger if logger else logging.getLogger(self.__class__.__name__)

    def __repr__(self):
        status = "Empty" if self._data is None else f"{self._data.shape}"
        return f"<{self.__class__.__name__}: {status}>"

    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Phương thức trừu tượng cần được implement bởi các lớp con.
        Thực hiện logic xử lý chính trên DataFrame.
        """
        pass

    def get_data(self):
        """Trả về DataFrame hiện tại đang được lưu trữ trong object."""
        return self._data

    # --- Các hàm tiện ích dùng chung (Utilities) ---
    def custom_text_mapping(self, column: str, mapping_dict: dict, new_column: str = None):
        """
        Áp dụng chuyển đổi text theo từ điển (Dict Mapping).
        Thường dùng để chuẩn hóa các cột Category (vd: 'Nội thất full' -> 'Full').
        """
        if column not in self._data.columns:
            self.logger.warning(f"Cột '{column}' không tồn tại để custom mapping.")
            return self

        target_col = new_column if new_column else column
        try:
            data_to_map = self._data[column].astype(str).str.strip()
            self._data[target_col] = data_to_map.map(mapping_dict)
            self.logger.info(f"Đã map dữ liệu cột {column} -> {target_col}.")
            
            # (Optional) Giữ lại giá trị cũ nếu không map được
            # self._data[target_col].fillna(self._data[column], inplace=True)
            
        except Exception as e:
            self.logger.error(f"Lỗi custom mapping cột {column}: {e}")
        return self

    def rename_columns(self, column_mapping: dict):
        """Đổi tên cột theo dictionary mapping."""
        if self._data is not None:
            self._data.rename(columns=column_mapping, inplace=True)
            self.logger.info(f"Đã đổi tên cột: {column_mapping}")
        return self

    def drop_columns(self, columns: list):
        """Xóa các cột không cần thiết khỏi DataFrame."""
        if self._data is not None:
            cols_to_drop = [c for c in columns if c in self._data.columns]
            if cols_to_drop:
                self._data.drop(columns=cols_to_drop, inplace=True)
                self.logger.info(f"Đã xóa cột: {cols_to_drop}")
        return self

# =============================================================================
# 2. DATA LOADER
# =============================================================================
class DataLoader:
    """
    Class tiện ích (Utility) chuyên trách việc Đọc/Ghi dữ liệu (I/O).
    Hỗ trợ các định dạng phổ biến: CSV, Excel, JSON.
    """
    @staticmethod
    def load_data(file_path: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
        log = logger if logger else logging.getLogger("DataLoader")
        
        if not os.path.exists(file_path):
            log.error(f"File không tồn tại: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Định dạng file chưa được hỗ trợ.")
            
            log.info(f"Đã đọc dữ liệu từ {file_path} - Shape: {df.shape}")
            return df
        except Exception as e:
            log.exception(f"Lỗi đọc file: {e}")
            return pd.DataFrame()

    @staticmethod
    def save_data(data: pd.DataFrame, output_path: str, logger: Optional[logging.Logger] = None):
        log = logger if logger else logging.getLogger("DataLoader")
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            if output_path.endswith('.csv'):
                data.to_csv(output_path, index=False)
            elif output_path.endswith('.xlsx'):
                data.to_excel(output_path, index=False)
            log.info(f"Đã lưu dữ liệu xuống: {output_path}")
        except Exception as e:
            log.error(f"Không thể lưu file: {e}")

# =============================================================================
# 3. MISSING VALUE IMPUTER
# =============================================================================
class MissingValueImputer(DataProcessor):
    """
    Chuyên trách xử lý dữ liệu thiếu (Missing Values).
    Hỗ trợ các chiến lược: mean, median, mode, ffill, constant, drop.
    """
    def __init__(self, strategy_map: Dict[str, str] = None, fill_value: Any = None, logger: logging.Logger = None):
        super().__init__(logger)
        self.strategy_map = strategy_map if strategy_map else {}
        self.fill_value = fill_value

    def process(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        self._data = data.copy()
        self.logger.info("Bắt đầu xử lý Missing Values...")
        
        strategies = kwargs.get('strategy_map', self.strategy_map)
        
        for col, strategy in strategies.items():
            if col not in self._data.columns: continue
            if self._data[col].isna().sum() == 0: continue

            try:
                if strategy == 'mean':
                    val = self._data[col].mean()
                    self._data[col] = self._data[col].fillna(val)
                elif strategy == 'median':
                    val = self._data[col].median()
                    self._data[col] = self._data[col].fillna(val)
                elif strategy == 'mode':
                    val = self._data[col].mode()[0]
                    self._data[col] = self._data[col].fillna(val)
                elif strategy == 'ffill':
                    self._data[col] = self._data[col].ffill()
                elif strategy == 'constant' and self.fill_value is not None:
                    self._data[col] = self._data[col].fillna(self.fill_value)
                elif strategy == 'drop':
                    self._data.dropna(subset=[col], inplace=True)
                
                self.logger.info(f"Col '{col}': Đã xử lý missing bằng chiến lược '{strategy}'.")
            except Exception as e:
                self.logger.error(f"Lỗi xử lý missing cột {col}: {e}")
        
        return self._data

# =============================================================================
# 4. OUTLIER HANDLER
# =============================================================================
class OutlierHandler(DataProcessor):
    """
    Chuyên trách xử lý ngoại lai (Outliers).
    Hỗ trợ phương pháp: IQR, Z-Score, Isolation Forest.
    Hành động: Xóa dòng (remove) hoặc Thay thế bằng NaN (replace_nan).
    """
    def __init__(self, columns: List[str], method: str = 'iqr', action: str = 'remove', threshold: float = 1.5, logger: logging.Logger = None):
        super().__init__(logger)
        self.columns = columns
        self.method = method
        self.action = action
        self.threshold = threshold

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self._data = data.copy()
        self.logger.info(f"Xử lý ngoại lai (Method: {self.method}, Action: {self.action})...")

        try:
            if self.method == 'isolation_forest':
                valid_cols = [c for c in self.columns if c in self._data.columns and pd.api.types.is_numeric_dtype(self._data[c])]
                if valid_cols:
                    temp_data = self._data[valid_cols].fillna(self._data[valid_cols].median())
                    clf = IsolationForest(random_state=42, contamination=0.05)
                    preds = clf.fit_predict(temp_data)
                    mask = preds == -1
                    if self.action == 'remove':
                        self._data = self._data[~mask]
                        self.logger.info(f"IsolationForest: Đã xóa {mask.sum()} dòng ngoại lai.")
            else:
                for col in self.columns:
                    if col not in self._data.columns or not pd.api.types.is_numeric_dtype(self._data[col]): continue
                    
                    mask = None
                    if self.method == 'iqr':
                        Q1 = self._data[col].quantile(0.25)
                        Q3 = self._data[col].quantile(0.75)
                        IQR = Q3 - Q1
                        mask = (self._data[col] < (Q1 - self.threshold * IQR)) | (self._data[col] > (Q3 + self.threshold * IQR))
                    elif self.method == 'z-score':
                        mean = self._data[col].mean()
                        std = self._data[col].std()
                        if std > 0:
                            z_scores = ((self._data[col] - mean) / std).abs()
                            mask = z_scores > self.threshold

                    if mask is not None and mask.any():
                        if self.action == 'remove':
                            self._data = self._data[~mask]
                            self.logger.info(f"Col '{col}': Đã xóa {mask.sum()} dòng ngoại lai.")
                        elif self.action == 'replace_nan':
                            self._data.loc[mask, col] = np.nan
                            self.logger.info(f"Col '{col}': Đã thay {mask.sum()} ngoại lai bằng NaN.")

        except Exception as e:
            self.logger.error(f"Lỗi xử lý ngoại lai: {e}")
        
        return self._data

# =============================================================================
# 5. DATA SCALER
# =============================================================================
class DataScaler(DataProcessor):
    """
    Chuyên trách chuẩn hóa dữ liệu số (Scaling).
    Hỗ trợ: StandardScaler (Z-score) và MinMaxScaler (0-1).
    """
    def __init__(self, columns: List[str], method: str = 'standard', logger: logging.Logger = None):
        super().__init__(logger)
        self.columns = columns
        self.method = method
        self.scalers = {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self._data = data.copy()
        ScalerClass = StandardScaler if self.method == 'standard' else MinMaxScaler
        
        for col in self.columns:
            if col in self._data.columns and pd.api.types.is_numeric_dtype(self._data[col]):
                try:
                    scaler = ScalerClass()
                    self._data[col] = scaler.fit_transform(self._data[col].values.reshape(-1, 1)).flatten()
                    self.scalers[col] = scaler
                except Exception as e:
                    self.logger.error(f"Lỗi chuẩn hóa cột {col}: {e}")
        return self._data

# =============================================================================
# 6. CATEGORICAL ENCODER
# =============================================================================
class CategoricalEncoder(DataProcessor):
    """
    Chuyên trách mã hóa biến phân loại (Encoding).
    Hỗ trợ: One-Hot Encoding và Label Encoding.
    """
    def __init__(self, columns: List[str], method: str = 'onehot', logger: logging.Logger = None):
        super().__init__(logger)
        self.columns = columns
        self.method = method
        self.encoders = {}

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self._data = data.copy()
        for col in self.columns:
            if col not in self._data.columns: continue
            try:
                if self.method == 'label':
                    le = LabelEncoder()
                    self._data[col] = le.fit_transform(self._data[col].astype(str))
                    self.encoders[col] = le
                elif self.method == 'onehot':
                    dummies = pd.get_dummies(self._data[col], prefix=col, drop_first=True)
                    self._data = pd.concat([self._data, dummies], axis=1)
                    self._data.drop(columns=[col], inplace=True)
            except Exception as e:
                self.logger.error(f"Lỗi mã hóa cột {col}: {e}")
        return self._data

# =============================================================================
# 7. FEATURE ENGINEER (Tạo đặc trưng mới)
# =============================================================================
class DateTimeEngineer(DataProcessor):
    """Class chuyên biệt để xử lý ngày tháng."""
    def __init__(self, columns: List[str]):
        super().__init__()
        self.columns = columns

    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        self._data = data.copy()
        for col in self.columns:
            if col in self._data.columns:
                try:
                    self._data[col] = pd.to_datetime(self._data[col], errors='coerce')
                    self._data[f'{col}_year'] = self._data[col].dt.year
                    self._data[f'{col}_month'] = self._data[col].dt.month
                    self._data[f'{col}_day'] = self._data[col].dt.day
                    self.logger.info(f"Đã tách ngày tháng cho cột: {col}")
                except Exception as e:
                    self.logger.error(f"Lỗi DateTime cột {col}: {e}")
        return self._data
    
# =============================================================================
# 8. REAL ESTATE SPECIALIST (Class Đặc Thù)
# =============================================================================
class RealEstateFeatureExtractor(DataProcessor):
    """
    Class chuyên biệt xử lý dữ liệu Bất Động Sản (BĐS).
    Chức năng:
    - Trích xuất thông tin từ chuỗi văn bản (Regex): Giá, Diện tích, Số phòng.
    - Chuẩn hóa địa danh (Quận/Huyện).
    - Tạo các đặc trưng mới (Feature Engineering).
    """
    def __init__(self, 
                 district_mapping: Dict[str, str], 
                 unit_factors: Dict[str, float], 
                 logger: logging.Logger = None):
        """
        Args:
            district_mapping: Dict ánh xạ tên quận huyện (từ Config).
            unit_factors: Dict tỷ giá quy đổi tiền tệ (từ Config).
        """
        super().__init__(logger)
        self.district_mapping = district_mapping
        self.unit_factors = {k.lower(): float(v) for k, v in unit_factors.items()}

    # --- HELPER METHODS (Logic Parsing) ---
    def _parse_room_logic(self, value) -> float:
        """Trích xuất số phòng từ chuỗi. VD: '2 phòng' -> 2.0"""
        if pd.isna(value) or str(value).strip() == '':
            return np.nan
        st = str(value).lower().strip()
        if 'nhiều hơn 10' in st: return 11.0
        if 'nhiều hơn 6' in st: return 7.0
        match = re.search(r"(\d+)", st)
        if match:
            return float(match.group(1))
        return np.nan

    def _parse_area(self, value) -> float:
        """Trích xuất diện tích (m2) từ chuỗi."""
        if pd.isna(value) or str(value).strip() == '': return np.nan
        match = re.search(r"(\d+([.,]\d+)?)", str(value))
        if match:
            try:
                return float(match.group(0).replace(',', '.'))
            except: pass
        return np.nan

    def _parse_price(self, value, target_unit='triệu') -> float:
        """
        Trích xuất giá tiền và quy đổi về đơn vị chuẩn.
        Sử dụng unit_factors để xử lý 'Tỷ', 'Triệu', 'Nghìn'.
        """
        if pd.isna(value) or str(value).strip() == '': return np.nan
        st = str(value).lower().strip()
        match = re.search(r"(\d+([.,]\d+)?)", st)
        if not match: return np.nan
        val = float(match.group(0).replace(',', '.'))

        # Tìm đơn vị
        input_factor = None
        sorted_units = sorted(self.unit_factors.keys(), key=len, reverse=True)
        for unit in sorted_units:
            if unit in st:
                input_factor = self.unit_factors[unit]
                break
        
        # Heuristic fallback: Đoán đơn vị nếu thiếu (dựa trên độ lớn)
        if input_factor is None:
            max_factor = self.unit_factors.get('tỷ', 1000.0)
            mid_factor = self.unit_factors.get('triệu', 1.0)
            input_factor = max_factor if val <= 300 else mid_factor

        target_factor = self.unit_factors.get(target_unit.lower(), 1.0)
        return val * (input_factor / target_factor)

    def _extract_district(self, address: str) -> str:
        """Trích xuất Quận/Huyện từ địa chỉ dài."""
        if not isinstance(address, str) or not address.strip(): return 'Khác'
        addr_lower = address.lower()
        full_text = f" {addr_lower} "

        # Ưu tiên 1: Tách dấu phẩy và tìm từ đuôi lên
        parts = [p.strip() for p in addr_lower.split(',')]
        if len(parts) > 1:
            for part in reversed(parts[-3:]):
                for k, v in self.district_mapping.items():
                    if k in part: return v
        
        # Ưu tiên 2: Quét toàn chuỗi
        for k, v in self.district_mapping.items():
            if k in full_text: return v
        return 'Khác'

    def _standardize_type_logic(self, value):
        """Gom nhóm loại hình BĐS."""
        if pd.isna(value) or str(value).strip() == '': return 'Loại Khác'
        loai = str(value).lower().strip()
        if 'penthouse' in loai or 'duplex' in loai: return 'Loại Hạng Sang'
        if 'officetel' in loai or 'dịch vụ' in loai: return 'Loại Văn Phòng'
        if 'chung cư' in loai or 'căn hộ' in loai: return 'Loại Chung Cư'
        if 'tập thể' in loai or 'cư xá' in loai: return 'Loại Tập Thể'
        return 'Loại Khác'
    
    def _parse_mixed_price_area(self, value, target_currency='tỷ'):
        """
        Xử lý chuỗi hỗn hợp: "5.5 Tỷ - 80m2".
        Trả về Tuple (Giá, Diện tích).
        """
        if pd.isna(value) or str(value).strip() == '': return np.nan, np.nan
        st = str(value).lower()

        area_val = np.nan
        price_part = st

        # 1. Tách diện tích (tìm "- ... m2")
        match_area = re.search(r"(?:-)\s*(\d+(?:[.,]\d+)?)\s*m2", st)
        if match_area:
            raw_area = match_area.group(0)
            area_val = self._parse_area(raw_area) # Tái sử dụng logic area
            price_part = st.replace(raw_area, "").strip() # Xóa diện tích đi

        # 2. Xử lý phần còn lại là Giá
        price_val = self._parse_price(price_part, target_unit=target_currency) # Tái sử dụng logic price

        return price_val, area_val

    # --- MAIN ACTION METHODS (Được Main.py gọi) ---
    
    def extract_room_feature(self, col: str):
        """Làm sạch cột số phòng (Ngủ/Tắm)."""
        if col in self._data.columns:
            self._data[col] = self._data[col].apply(self._parse_room_logic)
            self.logger.info(f"Đã trích xuất số liệu phòng cho cột: {col}")
        else:
            self.logger.warning(f"Cột {col} không tồn tại để xử lý phòng.")
        return self

    def standardize_binary(self, col: str, true_label='True', false_label='False'):
        """Biến đổi cột thành nhị phân (Có dữ liệu -> True, NaN -> False)."""
        if col in self._data.columns:
            self._data[col] = np.where(self._data[col].isna(), false_label, true_label)
            self.logger.info(f"Đã chuẩn hóa nhị phân cột {col} (NaN={false_label})")
        return self

    def standardize_category(self, col: str, new_col: str = None):
        """Chuẩn hóa cột Loại BĐS theo logic nghiệp vụ."""
        if col in self._data.columns:
            target = new_col if new_col else f"{col}_Standard"
            self._data[target] = self._data[col].apply(self._standardize_type_logic)
            self.logger.info(f"Đã chuẩn hóa loại BĐS: {col} -> {target}")
        return self
    
    def extract_mixed_data(self, source_col: str, price_col_out: str, area_col_out: str, target_currency='tỷ'):
        """Tách cột hỗn hợp (Giá - Diện tích) thành 2 cột riêng."""
        if source_col not in self._data.columns:
            self.logger.warning(f"Cột nguồn {source_col} không tồn tại.")
            return self

        self.logger.info(f"Đang tách dữ liệu hỗn hợp từ cột '{source_col}'...")
        try:
            # Dùng lambda để truyền target_currency vào hàm con
            tuples = self._data[source_col].apply(
                lambda x: self._parse_mixed_price_area(x, target_currency)
            ).to_list()

            # Unzip tuple thành 2 list
            prices, areas = zip(*tuples)

            self._data[price_col_out] = prices
            self._data[area_col_out] = areas

            count_p = self._data[price_col_out].notna().sum()
            count_a = self._data[area_col_out].notna().sum()
            self.logger.info(f"Tách thành công. Tìm thấy {count_p} giá, {count_a} diện tích.")
        except Exception as e:
            self.logger.error(f"Lỗi khi tách cột hỗn hợp: {e}")

        return self

    def cross_check_and_fix_area(self, area_col, price_col, unit_price_col, threshold=5.0):
        """
        Kiểm tra chéo: Diện tích = Tổng giá / Đơn giá.
        Nếu sai lệch vượt ngưỡng threshold -> Sửa lại Diện tích theo tính toán.
        """
        required = [area_col, price_col, unit_price_col]
        for c in required:
            if c not in self._data.columns:
                self.logger.warning(f"Thiếu cột {c}, bỏ qua kiểm tra chéo.")
                return self

        try:
            calc_area = self._data[price_col] / self._data[unit_price_col]
            ratio = self._data[area_col] / calc_area
            
            # Điều kiện lỗi: Sai lệch lớn hơn threshold VÀ tính toán ra số dương
            mask = (ratio.fillna(0) >= threshold) & (calc_area > 0)
            count = mask.sum()

            if count > 0:
                self.logger.info(f"Phát hiện {count} dòng sai diện tích (lệch > {threshold}x). Đang sửa...")
                self._data.loc[mask, area_col] = calc_area[mask]
            else:
                self.logger.info("Kiểm tra chéo: Dữ liệu hợp lý.")
        except Exception as e:
            self.logger.error(f"Lỗi kiểm tra chéo: {e}")
        return self

    def process(self, data: pd.DataFrame, 
                col_area='DienTich', 
                col_price='Gia', 
                col_addr='DiaChi', 
                target_currency='triệu') -> pd.DataFrame:
        """
        Hàm wrapper chạy các bước trích xuất cơ bản (Giá, Diện tích, Quận).
        """
        self._data = data.copy()
        self.logger.info(f"Trích xuất đặc trưng BĐS (Đơn vị đích: {target_currency})...")

        if col_area in self._data.columns:
            self._data[f'{col_area}_Clean'] = self._data[col_area].apply(self._parse_area)
            self.logger.info(f"Đã clean diện tích: {col_area}")

        if col_price in self._data.columns:
            self._data[f'{col_price}_Clean'] = self._data[col_price].apply(
                lambda x: self._parse_price(x, target_unit=target_currency)
            )
            self.logger.info(f"Đã clean giá: {col_price}")

        if col_addr in self._data.columns:
            self._data['Quan_Huyen'] = self._data[col_addr].apply(self._extract_district)
            self.logger.info(f"Đã trích xuất Quận/Huyện từ: {col_addr}")

        return self._data