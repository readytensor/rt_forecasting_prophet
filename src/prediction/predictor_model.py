import os
import sys
import warnings
from typing import List
from tqdm import tqdm

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
    made_up_frequency = "S"  # by seconds
    made_up_start_dt = "2000-01-01 00:00:00"

    def __init__(
        self,
        id_col,
        target_col,
        time_col,
        time_col_dtype,
        additional_regressors,
        growth="linear",
        seasonality_mode="multiplicative",
        uncertainty_samples=None,
        interval_width=95,
        run_type="multi",
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
        self.time_col_dtype = time_col_dtype
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
        self.time_to_int_map = {}
        self.all_ids = None

        self.multi_min_count = 5
        self.max_cpus_to_use = 6

    def prepare_data(self, data: pd.DataFrame, is_train=True) -> pd.DataFrame:
        """
        Function to prepare the dataframe to use with Prophet.

        Prophet expects the following columns:
        - ds: datetime column
        - y: target column of numeric type (float or int)

        If the time column is of type int, we will update it to be datetime
        by creating artificial dates starting at '1/1/2023 00:00:00'
        that increment by 1 second for each row.

        The final dataframe will have the following columns: 
        - <series_id>: name of column is kept as-is
        - ds: contains the datetime column. When passed integers, these are changed to
                datetimes as described above. We store the mapping of datetimes to 
                original integer values. 
                When passed date or datetimes, these are converted to datetime type
                and local time zone information is stripped (as per Prophet's 
                requirements.)
        - y: contains the target series to forecasting
        
        Additionally, there may be 0, 1 or more future covariates as were originally
        passed. These are returned as-is.  
        """
        # sort data
        data = data.sort_values(by=[self.id_col, self.time_col])

        if self.time_col_dtype == "INT":
            # Find the number of rows for each location (assuming all locations have
            # the same number of rows)
            series_val_counts = data[self.id_col].value_counts()
            series_len = series_val_counts.iloc[0]
            num_series = series_val_counts.shape[0]

            if is_train:
                # since prophet requires a datetime column, we will make up a timeline
                start_date = pd.Timestamp(self.made_up_start_dt)
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
                self.last_timestamp = datetimes[-1]
                self.timedelta = datetimes[-1] - datetimes[-2]
            else:
                start_date = self.last_timestamp + self.timedelta
                datetimes = pd.date_range(
                    start=start_date, periods=series_len, freq=self.made_up_frequency
                )
            int_vals = sorted(data[self.time_col].unique().tolist())
            self.time_to_int_map = dict(zip(datetimes, int_vals))
            # Repeat the datetime range for each location
            data[self.time_col] = list(datetimes) * num_series
        else:
            data[self.time_col] = pd.to_datetime(data[self.time_col])
            data[self.time_col] = data[self.time_col].dt.tz_localize(None)

        # rename columns as expected by Prophet
        data = data.rename(columns={self.target_col: "y", self.time_col: "ds"})
        reordered_cols = [self.id_col, "ds"]
        other_cols = [c for c in data.columns if c not in reordered_cols]
        reordered_cols.extend(other_cols)
        data = data[reordered_cols]
        return data

    def fit(self, history: pd.DataFrame) -> None:
        """Fit the Forecaster to the training data.
        A separate Prophet model is fit to each series that is contained
        in the data.

        Args:
            data (pandas.DataFrame): The features of the training data.
        """
        np.random.seed(0)
        history = self.prepare_data(history.copy())

        groups_by_ids = history.groupby(self.id_col)
        all_ids = list(groups_by_ids.groups.keys())
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.id_col) for id_ in all_ids
        ]

        self.models = {}
        if self.run_type == "sequential" or len(all_ids) <= self.multi_min_count:
            for id_, series in tqdm(zip(all_ids, all_series)):
                model = self._fit_on_series(history=series)
                self.models[id_] = model
        elif self.run_type == "multi":
            # Spare 2 cpus if we have many, but use at least 1 and no more than 6.
            cpus = max(1, min(cpu_count() - 2, self.max_cpus_to_use))
            print(f"Multi training with {cpus=}")
            p = Pool(cpus)
            models = list(tqdm(p.imap(self._fit_on_series, all_series)))

            for i, id_ in enumerate(all_ids):
                self.models[id_] = models[i]
        else:
            raise ValueError(
                f"Unrecognized run_type {self.run_type}. "
                "Must be one of ['sequential', 'multi']"
            )

        self.all_ids = all_ids
        self._is_trained = True

    def _fit_on_series(self, history: pd.DataFrame):
        """Prophet data model to given individual series of data"""
        model = Prophet(
            growth=self.growth,
            uncertainty_samples=self.uncertainty_samples,
            interval_width=self.interval_width,
            seasonality_mode=self.seasonality_mode,
        )
        for regressor in self.additional_regressors:
            model.add_regressor(regressor)
        model.fit(history)
        return model

    def predict(self, test_data: pd.DataFrame, prediction_col_name: str) -> np.ndarray:
        """Make the forecast of given length.

        Args:
            test_data (pd.DataFrame): Given test input for forecasting.
            prediction_col_name (str): Name to give to prediction column.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        future_df = self.prepare_data(test_data.copy(), is_train=False)
        groups_by_ids = future_df.groupby(self.id_col)
        all_series = [
            groups_by_ids.get_group(id_).drop(columns=self.id_col)
            for id_ in self.all_ids
        ]
        # for some reason, multi-processing takes longer! So use single-threaded.
        # forecast one series at a time
        all_forecasts = []
        for id_, series_df in tqdm(zip(self.all_ids, all_series)):
            forecast = self._predict_on_series(key_and_future_df=(id_, series_df))
            forecast.insert(0, self.id_col, id_)
            all_forecasts.append(forecast)

        # concatenate all series' forecasts into a single dataframe
        all_forecasts = pd.concat(all_forecasts, axis=0, ignore_index=True)
        all_forecasts["yhat"] = all_forecasts["yhat"].round(4)
        all_forecasts.rename(
            columns={
                "yhat": prediction_col_name,
                "ds": self.time_col,
            },
            inplace=True,
        )
        # Change datetime back to integer
        if self.time_col_dtype == "INT":
            all_forecasts[self.time_col] = all_forecasts[self.time_col].map(
                self.time_to_int_map
            )
        return all_forecasts

    def _predict_on_series(self, key_and_future_df):
        """Make forecast on given individual series of data"""
        key, future_df = key_and_future_df
        if self.models.get(key) is not None:
            forecast = self.models[key].predict(future_df)
            df_cols_to_use = ["ds", "yhat", "yhat_lower", "yhat_upper"]
            cols = [c for c in df_cols_to_use if c in forecast.columns]
            forecast = forecast[cols]
        else:
            # no model found - key wasnt found in history, so cant forecast for it.
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
        return f"Model name: {self.model_name}"


def train_predictor_model(
    history: pd.DataFrame,
    id_col: str,
    target_col: str,
    time_col: str,
    time_col_dtype: str,
    past_covariates: List[str],
    future_covariates: List[str],
    static_covariates: List[str],
    hyperparameters: dict,
) -> Forecaster:
    """
    Instantiate and train the predictor model.

    Args:
        history (pd.DataFrame): The history (training) data inputs.
        id_col (str): Name of id column in the history dataframe.
        target_col (str): Name of target column in the history dataframe.
        time_col (str): Name of time (epoch) column in the history dataframe.
        time_col_dtype (str): Data type of time (epoch) column.
        past_covariates (List[str]): List of past covariates.
        static_covariates (List[str]): List of future covariates.
        static_covariates (List[str]): List of static covariates.
        hyperparameters (dict): Hyperparameters for Prophet

    Returns:
        'Forecaster': The Forecaster model
    """
    model = Forecaster(
        id_col=id_col,
        target_col=target_col,
        time_col=time_col,
        time_col_dtype=time_col_dtype,
        additional_regressors=future_covariates+static_covariates,
        **hyperparameters,
    )
    model.fit(history=history)
    return model


def predict_with_model(
    model: Forecaster, test_data: pd.DataFrame, prediction_col_name: str
) -> pd.DataFrame:
    """
    Make forecast.

    Args:
        model (Forecaster): The Forecaster model.
        test_data (pd.DataFrame): The test input data for forecasting.
        prediction_col_name (int): Name to give to prediction column.

    Returns:
        pd.DataFrame: The forecast.
    """
    return model.predict(test_data, prediction_col_name)


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
