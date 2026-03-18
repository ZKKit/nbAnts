# XGBoost/LightGBM training and prediction – regression, feature selection, early stopping, ensemble
# Now stores feature names used during training for prediction alignment.
# Supports GPU acceleration via 'device' parameter.

# src/model.py (updated with robust GPU fallback)

# src/model.py (updated with verbose=0 in XGBoost fit calls)

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import logging
from datetime import timedelta
import joblib

logger = logging.getLogger(__name__)

class TradingModel:
    def __init__(self, config):
        self.config = config
        self.model_type = config['model']['type']          # 'xgboost', 'lightgbm', or 'ensemble'
        self.target_horizon = int(config['model']['target_horizon'])
        self.classification = config['model']['classification']   # False for regression
        self.train_window = int(config['model']['train_window'])
        self.retrain_freq = int(config['model']['retrain_frequency'])
        self.confidence_threshold = float(config['model'].get('confidence_threshold', 0.0))
        self.use_feature_selection = config['model'].get('use_feature_selection', False)
        self.early_stopping = config['model'].get('early_stopping_rounds', None)
        if self.early_stopping is not None:
            self.early_stopping = int(self.early_stopping)

        # Device selection
        self.device = config['model'].get('device', 'cpu')

        hp = config['model']['hyperparameters']
        if self.model_type == 'xgboost':
            self.hyperparams = hp['xgboost'].copy()
        elif self.model_type == 'lightgbm':
            self.hyperparams = hp['lightgbm'].copy()
        elif self.model_type == 'ensemble':
            self.hyperparams = {
                'xgb': hp['xgboost'].copy(),
                'lgb': hp['lightgbm'].copy()
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Inject device into hyperparameters (will be overridden later if needed)
        if self.device != 'cpu':
            if self.model_type == 'xgboost':
                self.hyperparams['device'] = self.device
            elif self.model_type == 'lightgbm':
                self.hyperparams['device'] = self.device
            elif self.model_type == 'ensemble':
                self.hyperparams['xgb']['device'] = self.device
                self.hyperparams['lgb']['device'] = self.device

        self.model = None          # will hold trained model(s)
        self.selected_features = None
        self.training_features = None   # list of feature names used in training

        # Flags for GPU fallback
        self._lgb_cuda_available = True   # assume CUDA works initially

    def prepare_features_target(self, df):
        """Create target (forward return) and separate features. Drops rows with NaN target."""
        df = df.copy()
        # target: future return over target_horizon days
        df['target'] = df.groupby('symbol')['close'].pct_change(self.target_horizon).shift(-self.target_horizon)
        if self.classification:
            df['target'] = (df['target'] > 0).astype(int)
        df.dropna(subset=['target'], inplace=True)

        exclude = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'target']
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols]
        y = df['target']
        return X, y

    def prepare_features(self, df):
        """Prepare features only (no target). Used for prediction – keeps all rows."""
        df = df.copy()
        exclude = ['open', 'high', 'low', 'close', 'volume', 'symbol', 'target']
        feature_cols = [c for c in df.columns if c not in exclude]
        X = df[feature_cols]
        return X

    def train(self, X, y, X_val=None, y_val=None):
        """Train a model with optional validation for early stopping."""
        # Store the feature names used in training
        self.training_features = X.columns.tolist()

        eval_set = None
        use_early_stop = False
        if X_val is not None and y_val is not None and len(X_val) > 0:
            eval_set = [(X_val, y_val)]
            use_early_stop = True

        early_stopping = self.early_stopping if use_early_stop else None

        # --- Feature selection (single model only) ---
        if self.use_feature_selection and self.model_type != 'ensemble':
            # Train a preliminary model to get importances
            if self.model_type == 'xgboost':
                prelim = xgb.XGBRegressor(**self.hyperparams, verbosity=0, random_state=42) \
                         if not self.classification else \
                         xgb.XGBClassifier(**self.hyperparams, verbosity=0, random_state=42)
            else:  # lightgbm
                prelim = lgb.LGBMRegressor(**self.hyperparams, verbose=-1, random_state=42) \
                         if not self.classification else \
                         lgb.LGBMClassifier(**self.hyperparams, verbose=-1, random_state=42)

            prelim.fit(X, y)

            selector = SelectFromModel(prelim, threshold='median', prefit=True)
            selected_mask = selector.get_support()
            self.selected_features = X.columns[selected_mask].tolist()
            logger.info(f"Selected {len(self.selected_features)} / {X.shape[1]} features")

            X_selected = X[self.selected_features]
            X_val_selected = X_val[self.selected_features] if X_val is not None else None
            eval_set_sel = [(X_val_selected, y_val)] if X_val_selected is not None else None

            # Train final model on selected features
            if self.model_type == 'xgboost':
                params = self.hyperparams.copy()
                if early_stopping:
                    params['early_stopping_rounds'] = early_stopping
                final = xgb.XGBRegressor(**params, verbosity=0, random_state=42) \
                        if not self.classification else \
                        xgb.XGBClassifier(**params, verbosity=0, random_state=42)
                final.fit(X_selected, y, eval_set=eval_set_sel, verbose=0)   # <-- ADDED verbose=0
            else:
                params = self.hyperparams.copy()
                if early_stopping:
                    params['early_stopping_rounds'] = early_stopping
                final = lgb.LGBMRegressor(**params, verbose=-1, random_state=42) \
                        if not self.classification else \
                        lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
                final.fit(X_selected, y, eval_set=eval_set_sel)

            self.model = final

        else:
            # No feature selection or ensemble
            if self.model_type == 'xgboost':
                params = self.hyperparams.copy()
                if early_stopping:
                    params['early_stopping_rounds'] = early_stopping
                model = xgb.XGBRegressor(**params, verbosity=0, random_state=42) \
                        if not self.classification else \
                        xgb.XGBClassifier(**params, verbosity=0, random_state=42)
                model.fit(X, y, eval_set=eval_set, verbose=0)   # <-- ADDED verbose=0

            elif self.model_type == 'lightgbm':
                params = self.hyperparams.copy()
                if early_stopping:
                    params['early_stopping_rounds'] = early_stopping
                # If GPU is requested but we know it's unavailable, remove device param
                if self.device in ('cuda', 'gpu') and not self._lgb_cuda_available:
                    params.pop('device', None)
                    logger.info("LightGBM GPU unavailable, using CPU.")
                try:
                    model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42) \
                            if not self.classification else \
                            lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
                    model.fit(X, y, eval_set=eval_set)
                except Exception as e:
                    err_str = str(e)
                    if self.device in ('cuda', 'gpu') and self._lgb_cuda_available and ('CUDA' in err_str or 'GPU' in err_str):
                        logger.warning(f"LightGBM GPU training failed: {e}. Falling back to CPU.")
                        self._lgb_cuda_available = False
                        params.pop('device', None)
                        model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42) \
                                if not self.classification else \
                                lgb.LGBMClassifier(**params, verbose=-1, random_state=42)
                        model.fit(X, y, eval_set=eval_set)
                    else:
                        raise

            else:  # ensemble
                # Train both models
                xgb_params = self.hyperparams['xgb'].copy()
                if early_stopping:
                    xgb_params['early_stopping_rounds'] = early_stopping
                xgb_model = xgb.XGBRegressor(**xgb_params, verbosity=0, random_state=42) \
                            if not self.classification else \
                            xgb.XGBClassifier(**xgb_params, verbosity=0, random_state=42)
                xgb_model.fit(X, y, eval_set=eval_set, verbose=0)   # <-- ADDED verbose=0

                # LightGBM part with GPU fallback logic
                lgb_params = self.hyperparams['lgb'].copy()
                if early_stopping:
                    lgb_params['early_stopping_rounds'] = early_stopping
                # If GPU is requested but we know it's unavailable, remove device param
                if self.device in ('cuda', 'gpu') and not self._lgb_cuda_available:
                    lgb_params.pop('device', None)
                    logger.info("LightGBM GPU unavailable, using CPU.")
                try:
                    lgb_model = lgb.LGBMRegressor(**lgb_params, verbose=-1, random_state=42) \
                                if not self.classification else \
                                lgb.LGBMClassifier(**lgb_params, verbose=-1, random_state=42)
                    lgb_model.fit(X, y, eval_set=eval_set)
                except Exception as e:
                    err_str = str(e)
                    if self.device in ('cuda', 'gpu') and self._lgb_cuda_available and ('CUDA' in err_str or 'GPU' in err_str):
                        logger.warning(f"LightGBM GPU training failed: {e}. Falling back to CPU.")
                        self._lgb_cuda_available = False
                        lgb_params.pop('device', None)
                        lgb_model = lgb.LGBMRegressor(**lgb_params, verbose=-1, random_state=42) \
                                    if not self.classification else \
                                    lgb.LGBMClassifier(**lgb_params, verbose=-1, random_state=42)
                        lgb_model.fit(X, y, eval_set=eval_set)
                    else:
                        raise

                self.model = (xgb_model, lgb_model)
                self.selected_features = None
                return

            self.model = model
            self.selected_features = None

    def predict(self, X):
        """Generate predictions (probabilities for classification, raw for regression)."""
        # Subset to training features if they are known (to handle extra columns)
        if self.training_features is not None:
            # Find intersection of training features and X columns
            common = [col for col in self.training_features if col in X.columns]
            if len(common) != len(self.training_features):
                missing = set(self.training_features) - set(X.columns)
                logger.warning(f"Missing features in prediction: {missing}. Filling with 0.")
                # Add missing columns with 0
                for col in missing:
                    X[col] = 0.0
            X = X[self.training_features]

        if self.selected_features is not None:
            X = X[self.selected_features]

        if self.model_type == 'ensemble':
            xgb_pred = self.model[0].predict_proba(X)[:, 1] if self.classification \
                       else self.model[0].predict(X)
            lgb_pred = self.model[1].predict_proba(X)[:, 1] if self.classification \
                       else self.model[1].predict(X)
            pred = (xgb_pred + lgb_pred) / 2
        else:
            if self.classification:
                pred = self.model.predict_proba(X)[:, 1]
            else:
                pred = self.model.predict(X)

        if self.classification:
            pred = np.where(pred >= self.confidence_threshold, pred, 0)
        return pred

    def walk_forward_predict(self, df, dates):
        """
        Walk-forward prediction: for each date in dates, train on preceding train_window days,
        validate on the most recent portion, then predict on that date.
        """
        predictions = []
        all_dates = sorted(dates)
        for i, current_date in enumerate(all_dates):
            # Only retrain on retrain_freq boundaries or if model is None
            if i % self.retrain_freq != 0 and self.model is not None:
                # Use existing model to predict
                pred_data = df.loc[current_date:current_date].copy()
                if pred_data.empty:
                    continue
                X_pred = self.prepare_features(pred_data)
                if X_pred.empty:
                    continue
                pred = self.predict(X_pred)
                pred_series = pd.Series(pred, index=pred_data.index, name='signal')
                predictions.append(pred_series)
                continue

            # Retrain: define training window
            train_end = current_date - timedelta(days=1)
            train_start = train_end - timedelta(days=self.train_window)
            train_data = df.loc[train_start:train_end].dropna()
            if train_data.empty:
                logger.warning(f"No training data for {current_date}, skipping")
                continue

            # Split into train/validation by time (last 20% of dates)
            train_dates = train_data.index.get_level_values(0).unique().sort_values()
            if len(train_dates) < 10:
                X_train, y_train = self.prepare_features_target(train_data)
                if X_train.empty:
                    continue
                self.train(X_train, y_train, X_val=None, y_val=None)
            else:
                split_idx = int(len(train_dates) * 0.8)
                train_cutoff = train_dates[split_idx]
                train_set = train_data.loc[:train_cutoff]
                val_set = train_data.loc[train_cutoff + timedelta(days=1):train_end]
                X_train, y_train = self.prepare_features_target(train_set)
                X_val, y_val = self.prepare_features_target(val_set) if not val_set.empty else (None, None)
                if X_train.empty:
                    continue
                self.train(X_train, y_train, X_val, y_val)

            # Predict on current date
            pred_data = df.loc[current_date:current_date].copy()
            if pred_data.empty:
                continue
            X_pred = self.prepare_features(pred_data)
            if X_pred.empty:
                continue
            pred = self.predict(X_pred)
            pred_series = pd.Series(pred, index=pred_data.index, name='signal')
            predictions.append(pred_series)

        if predictions:
            return pd.concat(predictions)
        else:
            return pd.Series(dtype=float)

    def update(self, X, y):
        """Online update (not used)."""
        raise NotImplementedError("Online update not implemented – use retraining.")

    def save_model(self, path):
        """Save model and metadata."""
        if self.model_type == 'ensemble':
            model_to_save = (self.model[0], self.model[1])
        else:
            model_to_save = self.model
        joblib.dump({
            'model_type': self.model_type,
            'model': model_to_save,
            'selected_features': self.selected_features,
            'training_features': self.training_features,
            'classification': self.classification,
            'confidence_threshold': self.confidence_threshold,
            'hyperparams': self.hyperparams,
            'target_horizon': self.target_horizon,
            'device': self.device
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load model and restore attributes."""
        data = joblib.load(path)
        self.model_type = data['model_type']
        self.model = data['model']
        self.selected_features = data.get('selected_features', None)
        self.training_features = data.get('training_features', None)
        self.classification = data['classification']
        self.confidence_threshold = data['confidence_threshold']
        self.hyperparams = data.get('hyperparams', {})
        self.target_horizon = data.get('target_horizon', 1)
        self.device = data.get('device', 'cpu')
        logger.info(f"Model loaded from {path}")
