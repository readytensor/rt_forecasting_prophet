import os
from typing import Dict, List, Tuple

import joblib

from data_models.schema_validator import validate_schema_dict
from utils import read_json_as_dict

SCHEMA_FILE_NAME = "schema.joblib"


class ForecastingSchema:
    """
    A class for loading and providing access to a forecaster schema.

    This class allows users to work with a generic schema for forecaster
    problems, enabling them to create algorithm implementations that are not hardcoded
    to specific feature names. The class provides methods to retrieve information about
    the schema, such as the ID field, target field, time field (if provided), and 
    exogenous fields (if provided). This makes it easier to preprocess and manipulate
    the input data according to the schema, regardless of the specific dataset used.
    """

    def __init__(self, schema_dict: dict) -> None:
        """
        Initializes a new instance of the `ForecastingSchema` class
        and using the schema dictionary.

        Args:
            schema_dict (dict): The python dictionary of schema.
        """
        self.schema = schema_dict
        self._exogenous_features = self._get_exogenous_features()

    @property
    def model_category(self) -> str:
        """
        Gets the model category.

        Returns:
            str: The category of the machine learning model
        """
        return self.schema["modelCategory"]

    @property
    def title(self) -> str:
        """
        Gets the title of the dataset or problem.

        Returns:
            str: The title of the dataset or the problem.
        """
        return self.schema["title"]

    @property
    def description(self) -> str:
        """
        Gets the description of the dataset or problem.

        Returns:
            str: A brief description of the dataset or the problem.
        """
        return self.schema["description"]

    @property
    def schema_version(self) -> float:
        """
        Gets the version number of the schema.

        Returns:
            float: The version number of the schema.
        """
        return self.schema["schemaVersion"]

    @property
    def input_data_format(self) -> str:
        """
        Gets the format of the input data.

        Returns:
            str: The format of the input data (e.g., CSV, JSON, etc.).
        """
        return self.schema["inputDataFormat"]

    @property
    def encoding(self) -> str:
        """
        Gets the encoding of the input data.

        Returns:
            str: The encoding of the input data (e.g., "utf-8", "iso-8859-1", etc.).
        """
        return self.schema["encoding"]

    @property
    def frequency(self) -> str:
        """
        Gets the frequency of the data.

        Returns:
            str: The frequency of the day.
        """
        return str(self.schema["frequency"])

    @property
    def forecast_length(self) -> int:
        """
        Gets the forecast_length of the data.

        Returns:
            int: The forecast_length of the data.
        """
        return int(self.schema["forecastLength"])

    @property
    def exogenous_features(self) -> List[str]:
        """
        Gets the exogenous features of the data.

        Returns:
            List[str]: The exogenous features list.
        """
        return self._exogenous_features

    def _get_exogenous_features(self) -> List[str]:
        """
        Returns the feature names of numeric and categorical data types.

        Returns:
            List[str]: The list of exogenous feature names.
        """
        if "additionalFeatures" not in self.schema:
            return []
        if len(self.schema["additionalFeatures"]) == 0:
            return []
        fields = self.schema["additionalFeatures"]
        exogenous_features = [f["name"] for f in fields if f["dataType"] == "NUMERIC"]
        return exogenous_features

    @property
    def id_col(self) -> str:
        """
        Gets the name of the ID field.

        Returns:
            str: The name of the ID field.
        """
        return self.schema["id"]["name"]

    @property
    def id_description(self) -> str:
        """
        Gets the description for the ID field.

        Returns:
            str: The description for the ID field.
        """
        return self.schema["id"].get(
            "description", "No description for target available."
        )

    @property
    def time_col(self) -> str:
        """
        Gets the name of the time field.

        Returns:
            str: The name of the ID field.
        """
        if "timeField" not in self.schema:
            return None
        return self.schema["timeField"]["name"]

    @property
    def time_description(self) -> str:
        """
        Gets the description for the ID field.

        Returns:
            str: The description for the ID field.
        """
        if "timeField" not in self.schema:
            return "No time field specified in schema"
        return self.schema["timeField"].get(
            "description", "No description for time field available."
        )

    @property
    def target(self) -> str:
        """
        Gets the name of the target field to forecast.

        Returns:
            str: The name of the target field.
        """
        return self.schema["forecastTarget"]["name"]

    @property
    def target_description(self) -> str:
        """
        Gets the description for the target field.

        Returns:
            str: The description for the target field.
        """
        return self.schema["forecastTarget"].get(
            "description", "No description for target available."
        )

    def get_description_for_feature(self, feature_name: str) -> str:
        """
        Gets the description for a single feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            str: The description for the specified feature.
        """
        field = self._get_field_by_name(feature_name)
        return field.get("description", "No description for feature available.")

    def get_example_value_for_feature(self, feature_name: str) -> List[str]:
        """
        Gets the example value for a single feature.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            List[str]: The example values for the specified feature.
        """
        return self._get_field_by_name(feature_name).get("example", 0.0)

    def _get_field_by_name(self, feature_name: str) -> dict:
        """
        Gets the field dictionary for a given feature name.

        Args:
            feature_name (str): The name of the feature.

        Returns:
            dict: The field dictionary for the feature.

        Raises:
            ValueError: If the feature is not found in the schema.
        """
        fields = self.schema["additionalFeatures"]
        for field in fields:
            if field["name"] == feature_name:
                return field
        raise ValueError(f"Feature '{feature_name}' not found in the schema.")


def load_json_data_schema(schema_dir_path: str) -> ForecastingSchema:
    """
    Load the JSON file schema into a dictionary, validate the schema dict for
    its correctness, and use the validated schema to instantiate the schema provider.

    Args:
    - schema_dir_path (str): Path from where to read the schema json file.

    Returns:
        ForecastingSchema: An instance of the ForecastingSchema.
    """
    schema_dict = read_json_as_dict(input_path=schema_dir_path)
    #validated_schema_dict = validate_schema_dict(schema_dict=schema_dict)
    data_schema = ForecastingSchema(schema_dict)
    return data_schema


def save_schema(schema: ForecastingSchema, save_dir_path: str) -> None:
    """
    Save the schema to a JSON file.

    Args:
        schema (ForecastingSchema): The schema to be saved.
        save_dir_path (str): The dir path to save the schema to.
    """
    if not os.path.exists(save_dir_path):
        os.makedirs(save_dir_path)
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    joblib.dump(schema, file_path)


def load_saved_schema(save_dir_path: str) -> ForecastingSchema:
    """
    Load the saved schema from a JSON file.

    Args:
        save_dir_path (str): The path to load the schema from.

    Returns:
        ForecastingSchema: An instance of the ForecastingSchema.
    """
    file_path = os.path.join(save_dir_path, SCHEMA_FILE_NAME)
    if not os.path.exists(file_path):
        print("no such file")
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")
    return joblib.load(file_path)
