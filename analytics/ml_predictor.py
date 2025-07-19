"""
ML-driven Code Quality Predictor for CodeConductor

Uses machine learning to predict code quality, success rates, and provide warnings.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import joblib
import sqlite3
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class CodeQualityPredictor:
    """ML-based predictor for code quality and success rates"""

    def __init__(self, db_path: str = "data/metrics.db"):
        self.db_path = db_path
        self.quality_model = None
        self.success_model = None
        self.text_vectorizer = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # Feature names for interpretability
        self.feature_names = [
            "prompt_length",
            "code_blocks",
            "prev_pass_rate",
            "complexity_avg",
            "reward_avg",
            "iteration_count",
            "model_source_score",
            "strategy_score",
        ]

    def extract_features(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> np.ndarray:
        """Extract features from prompt and context"""
        if context is None:
            context = {}

        # Text-based features
        prompt_length = len(prompt)
        code_blocks = prompt.count("```")

        # Historical features
        prev_pass_rate = context.get("prev_pass_rate", 0.5)
        complexity_avg = context.get("complexity_avg", 0.5)
        reward_avg = context.get("reward_avg", 30.0)
        iteration_count = context.get("iteration_count", 1)

        # Model and strategy features
        model_source = context.get("model_source", "mock")
        model_source_score = 1.0 if model_source == "lm_studio" else 0.5

        strategy = context.get("strategy", "conservative")
        strategy_scores = {"conservative": 0.8, "balanced": 0.6, "exploratory": 0.4}
        strategy_score = strategy_scores.get(strategy, 0.5)

        # Combine features
        features = np.array(
            [
                prompt_length / 1000.0,  # Normalize
                code_blocks / 10.0,
                prev_pass_rate,
                complexity_avg,
                reward_avg / 100.0,  # Normalize
                iteration_count / 10.0,
                model_source_score,
                strategy_score,
            ]
        )

        return features.reshape(1, -1)

    def load_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load training data from database"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Load metrics data
            query = """
            SELECT 
                prompt_length,
                code_blocks,
                prev_pass_rate,
                complexity,
                reward,
                iteration,
                model_source,
                strategy,
                pass_rate,
                blocked
            FROM metrics 
            WHERE prompt_length IS NOT NULL
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            if df.empty:
                logger.warning("No training data found in database")
                return None, None, None

            # Prepare features
            X = df[
                [
                    "prompt_length",
                    "code_blocks",
                    "prev_pass_rate",
                    "complexity",
                    "reward",
                    "iteration",
                ]
            ].values

            # Normalize features
            X = self.scaler.fit_transform(X)

            # Prepare targets
            y_quality = df["pass_rate"].values  # Regression target
            y_success = (df["pass_rate"] > 0.5).astype(int)  # Classification target

            logger.info(f"Loaded {len(df)} training samples")
            return X, y_quality, y_success

        except Exception as e:
            logger.error(f"Failed to load training data: {e}")
            return None, None, None

    def train_models(self) -> bool:
        """Train quality and success prediction models"""
        try:
            # Load data
            data = self.load_training_data()
            if data[0] is None:
                logger.warning("No training data available")
                return False

            X, y_quality, y_success = data

            # Split data
            (
                X_train,
                X_test,
                y_quality_train,
                y_quality_test,
                y_success_train,
                y_success_test,
            ) = train_test_split(
                X, y_quality, y_success, test_size=0.2, random_state=42
            )

            # Train quality prediction model (regression)
            self.quality_model = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.quality_model.fit(X_train, y_quality_train)

            # Train success prediction model (classification)
            self.success_model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
            self.success_model.fit(X_train, y_success_train)

            # Evaluate models
            quality_score = self.quality_model.score(X_test, y_quality_test)
            success_score = self.success_model.score(X_test, y_success_test)

            logger.info(f"Quality model R² score: {quality_score:.3f}")
            logger.info(f"Success model accuracy: {success_score:.3f}")

            self.is_trained = True
            return True

        except Exception as e:
            logger.error(f"Failed to train models: {e}")
            return False

    def predict_quality(
        self, prompt: str, context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Predict code quality and success probability"""
        if not self.is_trained or self.quality_model is None:
            logger.warning("Models not trained. Training now...")
            if not self.train_models():
                # Return mock prediction if training fails
                return {
                    "quality_score": 0.5,
                    "success_probability": 0.5,
                    "feature_importance": {},
                    "warnings": [
                        "⚠️ Using mock prediction - insufficient training data"
                    ],
                    "confidence": 0.3,
                    "timestamp": datetime.now().isoformat(),
                }

        try:
            # Extract features
            features = self.extract_features(prompt, context)

            # Check if scaler is fitted
            if not hasattr(self.scaler, "mean_"):
                # Use raw features if scaler not fitted
                features_scaled = features
            else:
                features_scaled = self.scaler.transform(features)

            # Make predictions
            quality_score = self.quality_model.predict(features_scaled)[0]
            success_prob = self.success_model.predict_proba(features_scaled)[0][1]

            # Get feature importance
            importance = self.quality_model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))

            # Generate warnings
            warnings = self._generate_warnings(quality_score, success_prob, context)

            return {
                "quality_score": float(quality_score),
                "success_probability": float(success_prob),
                "feature_importance": feature_importance,
                "warnings": warnings,
                "confidence": self._calculate_confidence(features_scaled[0]),
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return {"error": str(e)}

    def _generate_warnings(
        self, quality_score: float, success_prob: float, context: Dict[str, Any]
    ) -> List[str]:
        """Generate warnings based on predictions"""
        warnings = []

        # Quality warnings
        if quality_score < 0.3:
            warnings.append(
                "⚠️ Very low quality predicted - consider simplifying the prompt"
            )
        elif quality_score < 0.5:
            warnings.append("⚠️ Low quality predicted - review prompt complexity")

        # Success warnings
        if success_prob < 0.3:
            warnings.append("🚨 Low success probability - high risk of failure")
        elif success_prob < 0.5:
            warnings.append("⚠️ Moderate success probability - consider adjustments")

        # Strategy warnings
        strategy = context.get("strategy", "unknown")
        if strategy == "exploratory" and success_prob < 0.6:
            warnings.append("🔍 Exploratory strategy may be too risky for this prompt")

        # Historical warnings
        prev_pass_rate = context.get("prev_pass_rate", 0.5)
        if prev_pass_rate < 0.3:
            warnings.append(
                "📉 Historically low pass rate - consider different approach"
            )

        return warnings

    def _calculate_confidence(self, features: np.ndarray) -> float:
        """Calculate prediction confidence based on feature values"""
        # Simple confidence based on feature completeness
        non_zero_features = np.count_nonzero(features)
        total_features = len(features)
        confidence = non_zero_features / total_features

        # Boost confidence if we have good feature coverage
        if confidence > 0.8:
            confidence = min(confidence + 0.1, 1.0)

        return confidence

    def save_models(self, path: str = "models/"):
        """Save trained models"""
        try:
            Path(path).mkdir(parents=True, exist_ok=True)

            if self.quality_model:
                joblib.dump(self.quality_model, f"{path}/quality_model.pkl")
            if self.success_model:
                joblib.dump(self.success_model, f"{path}/success_model.pkl")
            if hasattr(self, "scaler"):
                joblib.dump(self.scaler, f"{path}/scaler.pkl")

            logger.info(f"Models saved to {path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save models: {e}")
            return False

    def load_models(self, path: str = "models/"):
        """Load trained models"""
        try:
            quality_path = f"{path}/quality_model.pkl"
            success_path = f"{path}/success_model.pkl"
            scaler_path = f"{path}/scaler.pkl"

            if Path(quality_path).exists():
                self.quality_model = joblib.load(quality_path)
            if Path(success_path).exists():
                self.success_model = joblib.load(success_path)
            if Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)

            self.is_trained = True
            logger.info("Models loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def get_trends(self, days: int = 30) -> Dict[str, Any]:
        """Analyze trends in code quality over time"""
        try:
            conn = sqlite3.connect(self.db_path)

            # Get recent data
            cutoff_date = datetime.now() - timedelta(days=days)
            query = """
            SELECT 
                DATE(timestamp) as date,
                AVG(pass_rate) as avg_pass_rate,
                AVG(complexity) as avg_complexity,
                AVG(reward) as avg_reward,
                COUNT(*) as iteration_count
            FROM metrics 
            WHERE timestamp > ?
            GROUP BY DATE(timestamp)
            ORDER BY date
            """

            df = pd.read_sql_query(query, conn, params=(cutoff_date.isoformat(),))
            conn.close()

            if df.empty:
                return {"error": "No data available for trend analysis"}

            # Calculate trends
            trends = {
                "pass_rate_trend": float(
                    df["avg_pass_rate"].iloc[-1] - df["avg_pass_rate"].iloc[0]
                ),
                "complexity_trend": float(
                    df["avg_complexity"].iloc[-1] - df["avg_complexity"].iloc[0]
                ),
                "reward_trend": float(
                    df["avg_reward"].iloc[-1] - df["avg_reward"].iloc[0]
                ),
                "total_iterations": int(df["iteration_count"].sum()),
                "avg_daily_iterations": float(df["iteration_count"].mean()),
                "improvement_rate": float(
                    df["avg_pass_rate"].iloc[-1] - df["avg_pass_rate"].iloc[0]
                )
                / len(df),
            }

            return trends

        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            return {"error": str(e)}


def create_predictor() -> CodeQualityPredictor:
    """Create and initialize predictor"""
    predictor = CodeQualityPredictor()

    # Try to load existing models
    if not predictor.load_models():
        # Train new models if loading fails
        predictor.train_models()
        predictor.save_models()

    return predictor


if __name__ == "__main__":
    # Test the predictor
    predictor = create_predictor()

    # Test prediction
    test_prompt = "Create a simple calculator function"
    test_context = {
        "strategy": "conservative",
        "prev_pass_rate": 0.7,
        "complexity_avg": 0.6,
        "reward_avg": 35.0,
        "iteration_count": 3,
        "model_source": "lm_studio",
    }

    result = predictor.predict_quality(test_prompt, test_context)
    print("Prediction result:", result)

    # Test trends
    trends = predictor.get_trends()
    print("Trends:", trends)
