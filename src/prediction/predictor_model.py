import os
import warnings
from typing import Optional, List
from tqdm import tqdm
import sys

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from multiprocessing import Pool, cpu_count
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Forecaster:
    """A wrapper class for the Prophet Forecaster.

    This class provides a consistent interface that can be used with other
    Forecaster models.
    """

    model_name = "Prophet Forecaster"

    def __init__(
        self,
        id_col,
        target_col,
        time_col,
        additional_regressors,
        growth="linear",
        seasonality_mode="additive",
        uncertainty_samples=None,
        interval_width=95,
        run_type="multi"
    ):
        """Construct a new Adaboost Forecaster.

        Args:
            n_estimators (int, optional): The maximum number of estimators
                at which boosting is terminated.
                Defaults to 100.
            learning_rate (int, optional): Weight applied to each Forecaster
                at each boosting iteration. A higher learning rate increases
                the contribution of each Forecaster.
                Defaults to 1e-1.
        """
        self.id_col = id_col
        self.target_col = target_col
        self.time_col = time_col
        self.additional_regressors = additional_regressors
        self.growth = growth
        self.seasonality_mode = seasonality_mode
        self.uncertainty_samples = uncertainty_samples
        self.interval_width = interval_width
        self.run_type = run_type
        self.models = None
        self._is_trained = False
        self.last_timestamp = None
        self.timedelta = None

        self.multi_min_count = 5
        self.print_period = 5
        self.max_cpus_to_use = 6

    
    def process_datetime_column(self, data: pd.DataFrame):
        """
        Function to add a datetime column that increments by 1 minute for each row, 
        starting at '1/1/2023 00:00:00'
        """
        if 'ds' not in data.columns:

            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[self.id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            # Generate the datetime range starting from '1/1/2023 00:00:00', with the
            # total count being n times the unique locations
            start_date = pd.Timestamp('2023-01-01 00:00:00')
            datetimes = pd.date_range(
                start=start_date,
                periods=series_len * num_series,
                freq='T'
            )        
            # Repeat the datetime range for each location
            data['ds'] = pd.Series(datetimes).\
                repeat(data[self.id_col].nunique()/num_series).reset_index(drop=True)
        else:
            data['ds'] = pd.to_datetime(data['ds'])
            datetimes = sorted(data['ds'].unique().tolist())
        self.last_timestamp = datetimes[-1]
        self.timedelta = datetimes[-1] - datetimes[-2]
        return data


    def fit(self, data: pd.DataFrame) -> None:
        """Fit the Forecaster to the training data.

        Args:
            data (pandas.DataFrame): The features of the training data.
        """
        history = data.rename(
            columns={
                self.target_col: 'y',
                self.time_col: 'ds'
            }
        )
        history = self.process_datetime_column(history)
        reordered_cols = [self.id_col, 'ds', 'y']
        history = history[reordered_cols]
        
        groups_by_ids = history.groupby(self.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [groups_by_ids.get_group(id_).drop(columns=self.id_col)
                      for id_ in all_ids]

        self.models = {}
        if (self.run_type == "sequential" or len(all_ids) <= self.multi_min_count):
            for id_, series in tqdm(zip(all_ids, all_series)):        
                model = self._fit_on_series( history=series )
                self.models[id_] = model
        elif self.run_type == "multi":
            # Spare 2 cpus if we have many, but use at least 1 and no more than 6.
            cpus = max(1, min(cpu_count()-2, self.max_cpus_to_use))
            print(f"Multi training with {cpus=}" )
            p = Pool(cpus)
            models = list(tqdm(p.imap(self._fit_on_series, all_series)))            

            for i, id_ in enumerate(all_ids):
                self.models[id_] = models[i] 
        else:
            raise ValueError(
                f"Unrecognized run_type {self.run_type}. "
                "Must be one of ['sequential', 'multi']")

        self.all_ids = all_ids
        self._is_trained = True

    def _fit_on_series(self, history):
        model=Prophet(
            growth=self.growth,
            uncertainty_samples=self.uncertainty_samples,
            interval_width=self.interval_width,
            seasonality_mode=self.seasonality_mode,
        )
        model.fit(history)
        return model

    def create_future_dataframe(self, forecast_length):
        """Create a future dataframe for the forecast.

        Args:
            forecast_length (int): The length of forecast.
        Returns:
            pandas.DataFrame: The future dataframe.
        """
        future = pd.DataFrame()
        datetimes = pd.date_range(
                start=self.last_timestamp,
                periods=forecast_length,
                freq=self.timedelta
            )
        future['ds'] = datetimes
        return future

    def predict(self, forecast_length, prediction_col_name) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            forecast_length (int): The length of forecast.
            prediction_col_name (str): Name for prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        future = self.create_future_dataframe(forecast_length)

        # for some reason, multi-processing takes longer! So use single-threaded.
        all_forecasts = []
        for id_ in tqdm(self.all_ids):
            forecast = self._predict_on_series(key_and_future_df = (id_, future))
            forecast.insert(0, self.id_col, id_)
            all_forecasts.append(forecast)

        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)
        all_forecasts['yhat'] = all_forecasts['yhat'].round(4)
        all_forecasts.rename(columns={"yhat": prediction_col_name}, inplace=True)
        del all_forecasts['ds']
        return all_forecasts


    def _predict_on_series(self, key_and_future_df): 
        key, future_df = key_and_future_df
        if self.models.get(key) is not None:
            forecast = self.models[key].predict(future_df)
            df_cols_to_use = ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
            cols = [ c for c in df_cols_to_use if c in forecast.columns]
            forecast = forecast[cols]
        else:
            # no model found - indicative of key not being in the history, so cant forecast for it. 
            forecast = None
        return forecast

    def save(self, model_dir_path: str) -> None:
        """Save the Forecaster to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Forecaster":
        """Load the Forecaster from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Forecaster: A new instance of the loaded Forecaster.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (f"Model name: {self.model_name}")


def train_predictor_model(
        data: pd.DataFrame,
        id_col: str,
        target_col: str,
        time_col: str,
        exog_cols: List[str],
        hyperparameters: dict
    ) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        data (pd.DataFrame): The training data inputs.
        train_y (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the Forecaster.

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
        additional_regressors=exog_cols,
        **hyperparameters
    )
    model.fit(data=data)
    return model


def predict_with_model(
    model: Forecaster, forecast_length: int, prediction_col_name: str
) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        forecast_length (int): The forecast length.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        np.ndarray: The forecast.
    """
    return model.predict(forecast_length, prediction_col_name)


def save_predictor_model(model: Forecaster, predictor_dir_path: str) -> None:
    """
    Save the Forecaster model to disk.

    Args:
        model (Forecaster): The Forecaster model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Forecaster:
    """
    Load the Forecaster model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Forecaster: A new instance of the loaded Forecaster model.
    """
    return Forecaster.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Forecaster, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the Forecaster model and return the accuracy.

    Args:
        model (Forecaster): The Forecaster model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the Forecaster model.
    """
    return model.evaluate(x_test, y_test)
