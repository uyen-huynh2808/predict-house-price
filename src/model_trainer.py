import os
import json
from typing import Dict, Any, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import logging

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, VotingRegressor 
from sklearn.preprocessing import MinMaxScaler, StandardScaler # [NEW] Import Scaler
from sklearn.svm import SVR
import xgboost as xgb

import matplotlib.pyplot as plt
import seaborn as sns

from .utils import set_seed

class ModelTrainer:
    """
    Lớp chịu trách nhiệm toàn bộ quy trình mô hình học máy (Regression):

    1. Nạp dữ liệu đã chuẩn hóa (load_data).
    2. Chia train/test (split_data).
    3. [NEW] Scale dữ liệu sau khi chia tập (scale_data) để tránh Data Leakage.
    4. Huấn luyện và tối ưu tham số cho nhiều mô hình (train_model + optimize_params).
    5. Đánh giá mô hình, ghi lại kết quả (evaluate).
    6. Lưu & load mô hình bằng joblib (save_model, load_model).
    7. So sánh kết quả giữa các mô hình và vẽ/lưu biểu đồ (barplot) trong pipeline.

    Class này thiết kế cho bài toán Regression (RMSE, MAE, R2).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
        cols_to_scale: Optional[List[str]] = None, # [NEW] Nhận danh sách cột cần scale
    ) -> None:
        """
        Parameters
        ----------
        config : dict
            Cấu hình đọc từ config.yaml/config.ini.
        logger : logging.Logger, optional
            Logger dùng để ghi log.
        cols_to_scale : List[str], optional
            Danh sách các cột cần thực hiện Scaling (MinMax/Standard).
            Nếu None thì sẽ không scale trong Trainer.
        """
        self.config = config

        # ==== Đọc config ====
        data_cfg = config.get("data", {})
        paths_cfg = config.get("paths", {})

        self.data_path: str = data_cfg.get(
            "processed_path", "data/processed/df_model_ready.csv"
        )
        self.target_col: str = data_cfg.get("target_col", "target")
        self.test_size: float = float(data_cfg.get("test_size", 0.2))
        self.random_seed: int = int(data_cfg.get("random_seed", 42))

        self.models_dir: str = paths_cfg.get("models_dir", "models/saved_models")
        self.metrics_dir: str = paths_cfg.get("metrics_dir", "models/metrics")
        self.figures_dir: str = paths_cfg.get("figures_dir", "reports/figures")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)

        self.logger = logger or logging.getLogger("ModelTrainerLogger")

        # [NEW] Danh sách cột cần scale và Scaler object
        self.cols_to_scale = cols_to_scale
        # Mặc định dùng MinMaxScaler để giữ giá trị [0, 1] (hoặc đổi thành StandardScaler nếu muốn)
        self.scaler = MinMaxScaler() 

        # ==== Biến dữ liệu ====
        self.df: Optional[pd.DataFrame] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None

        # ==== Mô hình & kết quả ====
        # mapping: model_name -> (estimator, param_dist)
        self.models_and_params: Dict[str, Tuple[Any, Dict[str, List[Any]]]] = {}
        # model_name -> best_estimator
        self.best_models: Dict[str, Any] = {}
        # list metrics cho từng model
        self.results: List[Dict[str, Any]] = []

        # Reproducibility
        set_seed(self.random_seed)

        # Khởi tạo 3 mô hình mặc định
        self._init_default_models()

        self.logger.info("===== ModelTrainer khởi tạo thành công =====")

    # ========= PROPERTIES / STATICMETHODS ==================================

    @property
    def best_model_name(self) -> Optional[str]:
        """
        Trả về tên mô hình tốt nhất (theo RMSE nhỏ nhất) sau khi train.
        """
        if not self.results:
            return None
        results_df = pd.DataFrame(self.results)
        
        # [FIX] Kiểm tra tên cột RMSE (có thể là RMSE_Test hoặc RMSE)
        if "RMSE_Test" in results_df.columns:
            best_row = results_df.sort_values(by="RMSE_Test").iloc[0]
        elif "RMSE" in results_df.columns:
            best_row = results_df.sort_values(by="RMSE").iloc[0]
        else:
            return None
            
        return str(best_row["model"])

    @staticmethod
    def _ensure_file_exists(path: str) -> None:
        """
        Hàm tiện ích kiểm tra file tồn tại.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"File không tồn tại: {path}")

    # ========= KHỞI TẠO MÔI TRƯỜNG ===========================================

    def _init_default_models(self) -> None:
        """
        Khởi tạo các mô hình với bộ tham số CHỐNG OVERFITTING mạnh hơn.
        """
        # 1. Random Forest: Tăng min_samples_leaf để cành cây không quá nhỏ
        rf = RandomForestRegressor(random_state=self.random_seed, n_jobs=-1)
        rf_param_dist = {
            "n_estimators": [200, 300, 500],
            "max_depth": [10, 15, 20],        # Giảm độ sâu (trước là 30)
            "min_samples_split": [5, 10, 15], # Tăng lên để tránh chia nhỏ quá
            "min_samples_leaf": [4, 8, 12],   # Quan trọng: Lá phải có ít nhất 4-12 mẫu
            "max_features": ["sqrt", "log2"]  # Chỉ xem xét 1 phần features mỗi lần
        }

        # 2. XGBoost: Thêm Regularization (L1/L2) để phạt mô hình phức tạp
        xgb_model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=self.random_seed,
            n_jobs=-1,
        )
        xgb_param_dist = {
            "n_estimators": [500, 1000, 1500],
            "max_depth": [3, 5, 6],           # Giữ cây nông (Shallow trees)
            "learning_rate": [0.01, 0.02],    # Học cực chậm
            "subsample": [0.6, 0.7],          # Chỉ học 60-70% dữ liệu mỗi cây
            "colsample_bytree": [0.6, 0.7],   # Chỉ học 60-70% cột mỗi cây
            "reg_alpha": [0.1, 1.0, 10.0],    # L1 Regularization (Chống nhiễu)
            "reg_lambda": [0.1, 1.0, 10.0]    # L2 Regularization (Chống nhiễu)
        }

        # 3. SVR (Giữ nguyên hoặc bỏ qua nếu chạy lâu)
        svr_model = SVR(kernel="linear")
        svr_param_dist = {
            "C": [0.01, 0.1, 1],
            "epsilon": [0.01, 0.1],
        }

        self.models_and_params = {
            "RandomForest": (rf, rf_param_dist),
            "XGBoost": (xgb_model, xgb_param_dist),
            "SVR": (svr_model, svr_param_dist),
        }

    # ========= CORE METHODS ================================================

    def load_data(self) -> None:
        """
        Nạp dữ liệu đã chuẩn hóa từ file CSV (đã numeric, sẵn sàng train).
        """
        self._ensure_file_exists(self.data_path)
        self.df = pd.read_csv(self.data_path)

        if self.target_col not in self.df.columns:
            raise ValueError(
                f"target_col='{self.target_col}' không tồn tại trong dữ liệu."
            )

        self.logger.info(f"Đã load data từ {self.data_path} với shape {self.df.shape}")

    def split_data(self) -> None:
        """
        Chia dữ liệu thành train/test bằng train_test_split.
        """
        if self.df is None:
            raise RuntimeError("Data chưa được load. Hãy gọi load_data() trước.")

        X = self.df.drop(columns=[self.target_col]).copy()
        y = self.df[self.target_col].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_seed,
        )

        self.logger.info(
            f"Đã chia train/test: "
            f"Train={self.X_train.shape}, Test={self.X_test.shape}"
        )
    
    def scale_data(self) -> None:
        """
        [NEW] Hàm scale dữ liệu SAU KHI SPLIT để đảm bảo không bị Data Leakage.
        Nguyên tắc:
        - Fit Scaler trên tập TRAIN (để học Min/Max).
        - Transform tập TRAIN.
        - Transform tập TEST (dùng Min/Max của TRAIN).
        """
        if self.cols_to_scale and self.X_train is not None:
            self.logger.info(f"Đang thực hiện Scaling (MinMax) trên các cột: {self.cols_to_scale}")
            
            # Kiểm tra xem các cột có tồn tại không
            missing_cols = [c for c in self.cols_to_scale if c not in self.X_train.columns]
            if missing_cols:
                self.logger.warning(f"Các cột sau không tìm thấy để scale: {missing_cols}. Sẽ bỏ qua.")
                valid_cols = [c for c in self.cols_to_scale if c in self.X_train.columns]
            else:
                valid_cols = self.cols_to_scale

            if valid_cols:
                # 1. Fit & Transform trên Train
                self.X_train[valid_cols] = self.scaler.fit_transform(self.X_train[valid_cols])
                
                # 2. Chỉ Transform trên Test (Dùng tham số của Train áp lên Test)
                self.X_test[valid_cols] = self.scaler.transform(self.X_test[valid_cols])
                
                # 3. Lưu scaler lại để sau này dùng cho dữ liệu thực tế (Inference)
                scaler_path = os.path.join(self.models_dir, "scaler.joblib")
                joblib.dump(self.scaler, scaler_path)
                self.logger.info(f"Đã lưu Scaler tại: {scaler_path}")
            else:
                self.logger.warning("Không có cột nào hợp lệ để scale.")

    def optimize_params(
        self, model_name: str, model: Any, param_dist: Dict[str, List[Any]]
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Tối ưu siêu tham số cho một mô hình bằng RandomizedSearchCV.
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Dữ liệu train chưa được chuẩn bị. Hãy gọi split_data().")

        self.logger.info("=" * 60)
        self.logger.info(f"Tối ưu siêu tham số cho model: {model_name}")

        # [FIX] Đọc tham số từ Config (để không bị hardcode n_iter=4)
        training_cfg = self.config.get("training", {})
        n_iter = int(training_cfg.get("n_iter", 20))
        cv = int(training_cfg.get("cv", 3))
        
        self.logger.info(f"-> Chạy Search với n_iter={n_iter}, cv={cv}")

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter, # Sử dụng biến n_iter từ config
            cv=cv,         # Sử dụng biến cv từ config
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=self.random_seed,
            verbose=1,
        )

        search.fit(self.X_train, self.y_train)

        best_model = search.best_estimator_
        best_params = search.best_params_

        self.logger.info(f"Best params cho {model_name}: {best_params}")

        return best_model, best_params

    def evaluate(self, model_name: str, model: Any) -> Dict[str, Any]:
        """
        Đánh giá model trên tập test theo các metric Regression:
        MAE, MSE, RMSE, R2.
        """
        if self.X_test is None or self.y_test is None:
            raise RuntimeError(
                "Dữ liệu test chưa được chuẩn bị. Hãy gọi split_data() trước."
            )

        # 1. Đánh giá trên tập TEST
        y_pred = model.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(self.y_test, y_pred)

        # 2. Đánh giá trên tập TRAIN (để kiểm tra Overfitting)
        y_pred_train = model.predict(self.X_train)
        rmse_train = float(np.sqrt(mean_squared_error(self.y_train, y_pred_train)))
        r2_train = r2_score(self.y_train, y_pred_train)

        self.logger.info(
            f"[{model_name}] TRAIN: R2={r2_train:.4f}, RMSE={rmse_train:.4f} | "
            f"TEST: R2={r2:.4f}, RMSE={rmse:.4f}"
        )
        
        if r2_train - r2 > 0.15:
            self.logger.warning(f"-> Cảnh báo: [{model_name}] có dấu hiệu Overfitting!")

        metrics_dict = {
            "model": model_name,
            "MAE_Test": float(mae),
            "MSE_Test": float(mse),
            "RMSE_Test": float(rmse), # Đặt tên rõ là Test
            "R2_Test": float(r2),
            "RMSE_Train": float(rmse_train),
            "R2_Train": float(r2_train),
        }

        return metrics_dict

    def save_model(self, model: Any, model_name: str) -> str:
        """
        Lưu mô hình ra file .joblib.
        """
        file_name = f"{model_name}_best_model.joblib"
        model_path = os.path.join(self.models_dir, file_name)
        joblib.dump(model, model_path)
        self.logger.info(f"Đã lưu model [{model_name}] vào: {model_path}")
        return model_path

    def load_model(self, model_name: str) -> Any:
        """
        Load mô hình đã lưu bằng joblib.
        """
        file_name = f"{model_name}_best_model.joblib"
        model_path = os.path.join(self.models_dir, file_name)
        self._ensure_file_exists(model_path)
        model = joblib.load(model_path)
        self.logger.info(f"Đã load model [{model_name}] từ: {model_path}")
        return model

    # ========= HIGH-LEVEL PIPELINE ========================================

    def train_model(self) -> None:
        """
        Hàm high-level:

        - Lặp qua từng mô hình trong self.models_and_params.
        - Tối ưu siêu tham số (optimize_params).
        - Đánh giá (evaluate).
        - Lưu model.
        - Lưu kết quả thực nghiệm (CSV, JSON).
        - Vẽ & lưu biểu đồ so sánh RMSE, MAE.
        - Chọn và lưu mô hình tốt nhất (best_overall_model.joblib).
        """
        if self.X_train is None or self.y_train is None:
            raise RuntimeError("Hãy gọi load_data(), split_data() và scale_data() trước khi train.")

        self.results = []
        self.best_models = {}
        estimators_for_voting = []

        for model_name, (model, param_dist) in self.models_and_params.items():
            best_model, best_params = self.optimize_params(
                model_name=model_name, model=model, param_dist=param_dist
            )

            # Lưu best_model cho từng loại
            self.best_models[model_name] = best_model
            
            # Thu thập model cho Voting
            if model_name in ['RandomForest', 'XGBoost']:
                estimators_for_voting.append((model_name, best_model))

            # Evaluate + lưu metrics
            metrics_dict = self.evaluate(model_name, best_model)
            metrics_dict["best_params"] = best_params
            self.results.append(metrics_dict)

            # Save model riêng
            self.save_model(best_model, model_name)

        # Voting Regressor
        if len(estimators_for_voting) >= 2:
            self.logger.info("=" * 60)
            self.logger.info(f"Đang huấn luyện Voting Regressor từ: {[x[0] for x in estimators_for_voting]}...")
            voting_model = VotingRegressor(estimators=estimators_for_voting)
            voting_model.fit(self.X_train, self.y_train)
            
            v_metrics = self.evaluate("Voting_Ensemble", voting_model)
            v_metrics["best_params"] = "Ensemble (Soft Voting)"
            self.results.append(v_metrics)
            self.best_models["Voting_Ensemble"] = voting_model
            self.save_model(voting_model, "Voting_Ensemble")

        # Sau khi train xong tất cả:
        self._save_experiment_results()
        self._save_best_overall_model()
        self._plot_and_save_comparison_figures()

    # ========= LƯU KẾT QUẢ & HÌNH ========================================

    def _save_experiment_results(self) -> None:
        """
        Ghi lại các kết quả thực nghiệm vào CSV và JSON.
        """
        if not self.results:
            self.logger.warning("Không có kết quả nào để lưu.")
            return

        results_df = pd.DataFrame(self.results)

        csv_path = os.path.join(self.metrics_dir, "model_results_summary.csv")
        json_path = os.path.join(self.metrics_dir, "model_results_summary.json")

        results_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Đã lưu bảng kết quả vào: {csv_path}")
        self.logger.info(f"Đã lưu bảng kết quả JSON vào: {json_path}")

    def _save_best_overall_model(self) -> None:
        """
        Chọn mô hình tốt nhất theo RMSE và lưu riêng file 'best_overall_model.joblib'.
        """
        if not self.results:
            self.logger.warning("Không có kết quả để chọn best overall model.")
            return

        best_name = self.best_model_name
        if best_name is None:
            self.logger.warning("Không tìm được best model name.")
            return

        best_model = self.best_models.get(best_name)
        if best_model is None:
            self.logger.warning(
                f"Không tìm thấy best model tương ứng với tên: {best_name}"
            )
            return

        best_overall_path = os.path.join(self.models_dir, "best_overall_model.joblib")
        joblib.dump(best_model, best_overall_path)
        self.logger.info(
            f"Mô hình tốt nhất theo RMSE: {best_name}. "
            f"Đã lưu vào: {best_overall_path}"
        )

    def _plot_and_save_comparison_figures(self) -> None:
        """
        Vẽ và lưu biểu đồ so sánh RMSE, MAE giữa các mô hình
        (barplot) – phần này thuộc luôn vào class mô hình học máy.
        """
        if not self.results:
            self.logger.warning("Không có kết quả để vẽ biểu đồ.")
            return

        results_df = pd.DataFrame(self.results)
        
        # Kiểm tra tên cột
        y_rmse = "RMSE_Test" if "RMSE_Test" in results_df.columns else "RMSE"
        y_mae = "MAE_Test" if "MAE_Test" in results_df.columns else "MAE"

        # Barplot RMSE
        rmse_fig_path = os.path.join(self.figures_dir, "rmse_comparison.png")
        plt.figure(figsize=(8, 5))
        sns.barplot(data=results_df, x="model", y=y_rmse)
        plt.title("So sánh RMSE giữa các mô hình")
        plt.ylabel("RMSE")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(rmse_fig_path, dpi=300)
        plt.close()
        self.logger.info(f"Đã lưu biểu đồ RMSE: {rmse_fig_path}")

        # Barplot MAE
        mae_fig_path = os.path.join(self.figures_dir, "mae_comparison.png")
        plt.figure(figsize=(8, 5))
        sns.barplot(data=results_df, x="model", y=y_mae)
        plt.title("So sánh MAE giữa các mô hình")
        plt.ylabel("MAE")
        plt.xlabel("Model")
        plt.tight_layout()
        plt.savefig(mae_fig_path, dpi=300)
        plt.close()
        self.logger.info(f"Đã lưu biểu đồ MAE: {mae_fig_path}")

    # ========= HELPER METHODS FOR VISUALIZATION ============================

    def get_feature_names(self) -> List[str]:
        """
        Trả về danh sách tên các đặc trưng (columns) từ tập train.
        Hàm này hỗ trợ cho việc vẽ biểu đồ Feature Importance bên ngoài.
        """
        if self.X_train is not None:
            return self.X_train.columns.tolist()

        # Fallback: Nếu chưa split nhưng đã load data
        if self.df is not None:
             return self.df.drop(columns=[self.target_col]).columns.tolist()

        self.logger.warning("Chưa có dữ liệu để lấy feature names.")
        return []