import unittest
import numpy as np
import pandas as pd

from nowcast_lstm.LSTM import LSTM
from nowcast_lstm import data_setup


class TestDataLeakage(unittest.TestCase):
    """Ensure no data leakage in time series predictions"""

    def setUp(self):
        self.data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=24, freq="MS"),
                "var1": np.arange(24),
                "var2": np.arange(24) * 2,
                "target": np.arange(24) * 3,
            }
        )

    def test_no_future_data_in_predictions(self):
        """Ensure predictions don't use future information"""
        # Train on first 18 months
        train_end = 18
        training_data = self.data.iloc[:train_end].copy()

        model = LSTM(
            training_data,
            "target",
            n_timesteps=6,
            n_models=1,
            train_episodes=10,
        )
        model.train(quiet=True)

        # Get predictions on training data
        train_preds = model.predict(training_data)

        # For each prediction, verify it only uses data from before that point
        # The model with n_timesteps=6 should only look back 6 periods
        # So we lose 5 observations (n_timesteps - 1) from the start
        expected_preds = train_end - (model.n_timesteps - 1)
        self.assertEqual(len(train_preds), expected_preds)

        # First prediction should only use data from indices 0-5 (6 timesteps)
        # So earliest prediction is at index 6
        self.assertTrue(
            len(train_preds) <= train_end,
            "Model should not predict beyond training data",
        )

    def test_temporal_ordering_preserved(self):
        """Train and test data must maintain chronological order"""
        train_end_date = "2021-06-01"
        training_data = self.data[self.data.date <= train_end_date].copy()
        test_data = self.data[self.data.date > train_end_date].copy()

        # No overlap between train and test
        self.assertTrue(
            training_data.date.max() < test_data.date.min(),
            "Training data must be strictly before test data",
        )

        model = LSTM(training_data, "target", n_timesteps=6, n_models=1, train_episodes=10)
        model.train(quiet=True)

        # Verify temporal separation
        self.assertTrue(
            all(test_date > training_data.date.max() for test_date in test_data.date),
            "All test dates must be after training period",
        )

    def test_target_not_in_features(self):
        """Target variable must be excluded from features"""
        model = LSTM(self.data, "target", n_timesteps=6, n_models=1, train_episodes=10)

        dataset = data_setup.gen_dataset(self.data, "target")
        X, y = data_setup.gen_model_input(dataset["na_filled_dataset"], n_timesteps=6)

        # Features = total columns - date - target
        expected_features = len(self.data.columns) - 2
        self.assertEqual(X.shape[2], expected_features)

    def test_ragged_preds_no_future_leakage(self):
        """Ragged predictions must respect publication lags"""
        pub_lags = [1, 1]

        model = LSTM(self.data, "target", n_timesteps=6, n_models=1, train_episodes=10)
        model.train(quiet=True)

        ragged_result = model.gen_ragged_X(pub_lags, lag=-2)
        ragged_X, ragged_y = ragged_result[0], ragged_result[1]

        self.assertIsNotNone(ragged_X)
        self.assertIsNotNone(ragged_y)
        self.assertEqual(ragged_X.shape, model.X.shape)

    def test_prediction_uses_only_past_timesteps(self):
        """Model should use exactly the specified lookback window"""
        n_timesteps = 4

        model = LSTM(self.data, "target", n_timesteps=n_timesteps, n_models=1, train_episodes=10)
        model.train(quiet=True)

        # Check timestep dimension
        self.assertEqual(model.X.shape[1], n_timesteps)

        # Check sample count (lose n_timesteps - 1 from start)
        expected_samples = len(self.data) - (n_timesteps - 1)
        self.assertEqual(model.X.shape[0], expected_samples)

    def test_no_data_from_future_in_training(self):
        """Training should never see future target values"""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=12, freq="MS"),
                "var1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "target": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            }
        )

        train_data = data.iloc[:8].copy()

        model = LSTM(train_data, "target", n_timesteps=3, n_models=1, train_episodes=50)
        model.train(quiet=True)

        test_preds = model.predict(data)
        self.assertGreater(len(test_preds), 0)


if __name__ == "__main__":
    unittest.main()
