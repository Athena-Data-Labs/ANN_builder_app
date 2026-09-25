import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Use the same library import order as the app entry point.
import model_build
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest
from visualization.visualize import (
    fig_area_chart, plot_predicted_vs_actual, plot_cumulative_gain, cm_map,
)


# AppTest does not expose file upload interactions. Supply the CSV bytes while
# leaving the page, widgets, reruns and session state running normally.
UPLOADS = '''
import io
import streamlit as st

def uploaded_file(label, **kwargs):
    contents = st.session_state.get("test_uploads", {}).get(label)
    return io.BytesIO(contents) if contents is not None else None

st.file_uploader = uploaded_file
'''


def widget(widgets, label):
    return next(item for item in widgets if item.label == label)


class WorkflowTests(unittest.TestCase):
    def app(self, page):
        return AppTest.from_string(
            UPLOADS + f"\nfrom {page[0]} import {page[1]}\n{page[1]}()",
            default_timeout=30,
        )

    def model_app(self):
        at = self.app(("model_build", "build"))
        at.session_state["uploaded_df"] = pd.DataFrame({
            "target": [0, 1] * 20,
            "value": range(40),
            "category": [f"category_{i}" for i in range(40)],
        })
        at.run()
        self.assert_clean(at)
        return at

    def assert_clean(self, at):
        self.assertEqual([e.message for e in at.exception], [])
        self.assertEqual([e.value for e in at.error], [])

    def test_empty_pages(self):
        for page in (("model_build", "build"), ("data_preprocessing", "clean_preprocess"),
                     ("data_exploration", "data_exploration")):
            with self.subTest(page=page):
                self.assert_clean(self.app(page).run())

    def test_cleaning_survives_reruns_and_reset(self):
        at = self.app(("data_preprocessing", "clean_preprocess"))
        at.session_state["test_uploads"] = {"Upload a CSV file": b"x,y\n1,a\n1,a\n,b\n"}
        at.run()
        widget(at.button, "Apply Cleanup").click().run()
        self.assert_clean(at)
        cleaned = at.session_state["processed_df"].copy()
        self.assertEqual(len(cleaned), 2)
        self.assertFalse(cleaned.isna().any().any())
        at.run()
        pd.testing.assert_frame_equal(cleaned, at.session_state["processed_df"])
        widget(at.button, "Reset to original upload").click().run()
        self.assertEqual(len(at.session_state["processed_df"]), 3)
        self.assertTrue(at.session_state["processed_df"].isna().any().any())
        self.assert_clean(at)

    def test_drop_column_used_in_missing_subset(self):
        at = self.app(("data_preprocessing", "clean_preprocess"))
        at.session_state["test_uploads"] = {"Upload a CSV file": b"x,y\n1,a\n,b\n"}
        at.run()
        widget(at.multiselect, "Columns to remove").set_value(["x"])
        widget(at.radio, "Missing values strategy").set_value("Drop rows with missing values")
        widget(at.multiselect, "Drop rows only when these columns are missing").set_value(["x"])
        widget(at.button, "Apply Cleanup").click().run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["processed_df"].to_dict("list"), {"y": ["a"]})

    def test_empty_cleanup_does_not_discard_dataset(self):
        at = self.app(("data_preprocessing", "clean_preprocess"))
        at.session_state["test_uploads"] = {"Upload a CSV file": b"x,y\n1,a\n"}
        at.run()
        widget(at.multiselect, "Columns to remove").set_value(["x", "y"])
        widget(at.button, "Apply Cleanup").click().run()
        self.assert_clean(at)
        self.assertEqual(list(at.session_state["processed_df"].columns), ["x", "y"])
        self.assertTrue(any("no rows or columns" in w.value for w in at.warning))

    def test_split_and_transformer_survive_clicks(self):
        at = self.model_app()
        original = at.session_state["X_test"].copy()
        preprocessor = at.session_state["preprocessor"]
        widget(at.number_input, "Epochs").set_value(2).run()
        self.assert_clean(at)
        np.testing.assert_array_equal(original, at.session_state["X_test"])
        self.assertIs(preprocessor, at.session_state["preprocessor"])
        self.assertTrue(widget(at.button, "Run Model").disabled)

    def test_all_encoders_and_unseen_categories(self):
        at = self.model_app()
        for encoder in ("OneHotEncoder", "OrdinalEncoder", "TargetEncoder", "BinaryEncoder"):
            with self.subTest(encoder=encoder):
                widget(at.selectbox, "Select Encoder for Categorical Features").select(encoder).run()
                self.assert_clean(at)
                self.assertIn("X_train", at.session_state)
                self.assertEqual(at.session_state["X_train"].ndim, 2)
                result = at.session_state["preprocessor"].transform(
                    pd.DataFrame({"value": [41], "category": ["unseen"]})
                )
                self.assertTrue(np.isfinite(result).all())

    def test_data_and_settings_clear_trained_model(self):
        at = self.model_app()
        at.session_state["ann_model"] = object()
        at.session_state["prediction_results"] = pd.DataFrame({"old": [1]})
        widget(at.slider, "Select Test Size").set_value(0.4).run()
        self.assertNotIn("ann_model", at.session_state)
        self.assertNotIn("prediction_results", at.session_state)
        at.session_state["ann_model"] = object()
        data = at.session_state["uploaded_df"].copy()
        data.loc[0, "value"] = 100
        at.session_state["uploaded_df"] = data
        at.run()
        self.assertNotIn("ann_model", at.session_state)
        self.assert_clean(at)

    def test_invalid_model_settings_block_training(self):
        at = self.model_app()
        widget(at.number_input, "Number of Output Neurons").set_value(2).run()
        self.assertTrue(widget(at.button, "Build/Train Model").disabled)
        widget(at.number_input, "Number of Output Neurons").set_value(1).run()
        widget(at.radio, "Select Task Type").set_value("Regression").run()
        self.assertTrue(widget(at.button, "Build/Train Model").disabled)
        widget(at.selectbox, "Activation Function for Output Layer").select("linear").run()
        widget(at.selectbox, "Choose the Loss Function for Model Training").select("mean_squared_error").run()
        self.assertFalse(widget(at.button, "Build/Train Model").disabled)
        self.assert_clean(at)

    def test_bad_training_data_is_explained(self):
        for df in (pd.DataFrame({"target": [0, 1]}),
                   pd.DataFrame({"target": [0], "x": [1]}),
                   pd.DataFrame({"target": [0, 1], "x": [1, np.nan]}),
                   pd.DataFrame({"target": [0, 1], "x": [1, np.inf]}),
                   pd.DataFrame({"target": ["yes", "no"], "x": [1, 2]})):
            with self.subTest(df=df):
                at = self.app(("model_build", "build"))
                at.session_state["uploaded_df"] = df
                at.run()
                self.assert_clean(at)
                self.assertGreater(len(at.warning), 0)
                self.assertNotIn("ann_model", at.session_state)

    def test_use_cleaned_dataset(self):
        at = self.model_app()
        cleaned = pd.DataFrame({"target": [0, 1] * 10, "new_input": range(20)})
        at.session_state["processed_df"] = cleaned
        at.session_state["ann_model"] = object()
        at.run()
        widget(at.button, "Use cleaned dataset").click().run()
        self.assert_clean(at)
        pd.testing.assert_frame_equal(cleaned, at.session_state["uploaded_df"])
        self.assertEqual(at.session_state["X_columns"], ["new_input"])
        self.assertNotIn("ann_model", at.session_state)

    def test_network_visualization(self):
        at = self.model_app()
        widget(at.button, "Visualize Model").click().run()
        self.assert_clean(at)
        self.assertEqual(len(at.get("imgs")), 1)

    def test_replacing_uploaded_csv_clears_model_and_old_columns(self):
        at = self.model_app()
        at.session_state["ann_model"] = object()
        at.session_state["test_uploads"] = {
            "📥 Upload data for model training (CSV only):":
                b"new_target,new_input\n0,1\n1,2\n0,3\n1,4\n0,5\n1,6\n"
        }
        at.run()
        self.assert_clean(at)
        self.assertNotIn("ann_model", at.session_state)
        self.assertEqual(at.session_state["target"], "new_target")
        self.assertEqual(at.session_state["X_columns"], ["new_input"])

    def test_exploration_upload_becomes_active_cleaning_data(self):
        at = self.app(("data_exploration", "data_exploration"))
        at.session_state["processed_df"] = pd.DataFrame({"old": [1]})
        at.session_state["test_uploads"] = {
            "Please upload a CSV file:": b"new,flag\n1,True\n2,False\n"
        }
        at.run()
        widget(at.selectbox, "🎯 Select Target Variable").select("flag").run()
        self.assert_clean(at)
        self.assertEqual(list(at.session_state["processed_df"].columns), ["new", "flag"])

    def test_exploration_does_not_mutate_data_or_crash_on_text_target(self):
        at = self.app(("data_exploration", "data_exploration"))
        df = pd.DataFrame({"category": ["a", "b", "c"], "value": [0, 1, -2]})
        at.session_state["df"] = df.copy()
        at.run()
        widget(at.selectbox, "🎯 Select Target Variable").select("category").run()
        self.assert_clean(at)
        pd.testing.assert_frame_equal(df, at.session_state["df"])
        widget(at.selectbox, "🎯 Select Target Variable").select("value").run()
        self.assert_clean(at)

    def test_navigation_preserves_model_settings(self):
        at = AppTest.from_string(UPLOADS + "\nfrom main import main\nmain()", default_timeout=30)
        at.session_state["uploaded_df"] = pd.DataFrame({"x": range(40), "target": [0, 1] * 20})
        at.run()
        widget(at.selectbox, "Select a Page").select("🏗️ Build Artificial-Neural-Network").run()
        widget(at.selectbox, "Choose Target Feature").select("target").run()
        widget(at.number_input, "Epochs").set_value(1).run()
        model = object()
        at.session_state["ann_model"] = model
        original = at.session_state["X_test"].copy()
        widget(at.selectbox, "Select a Page").select("🤖 About this App").run()
        widget(at.selectbox, "Select a Page").select("🏗️ Build Artificial-Neural-Network").run()
        self.assert_clean(at)
        self.assertEqual(at.session_state["preprocessing"]["target"], "target")
        self.assertEqual(at.session_state["model_config"]["epochs"], 1)
        self.assertIs(at.session_state["ann_model"], model)
        np.testing.assert_array_equal(at.session_state["X_test"], original)

    def test_training_evaluation_prediction_and_download(self):
        for task in ("Binary Classification", "Regression"):
            with self.subTest(task=task):
                at = self.model_app()
                widget(at.number_input, "Number of Hidden Layers").set_value(1).run()
                widget(at.number_input, "Neurons in Layer 1").set_value(3).run()
                widget(at.number_input, "Epochs").set_value(1).run()
                if task == "Regression":
                    widget(at.radio, "Select Task Type").set_value(task).run()
                    widget(at.selectbox, "Activation Function for Output Layer").select("linear").run()
                    widget(at.selectbox, "Choose the Loss Function for Model Training").select("mean_squared_error").run()
                widget(at.button, "Build/Train Model").click().run(timeout=60)
                self.assert_clean(at)
                self.assertIn("ann_model", at.session_state)
                model = at.session_state["ann_model"]
                original = at.session_state["X_test"].copy()
                widget(at.button, "Run Model").click().run()
                self.assert_clean(at)
                self.assertIn("y_pred", at.session_state)
                np.testing.assert_array_equal(original, at.session_state["X_test"])
                at.session_state["test_uploads"] = {
                    "Upload new data for predictions:": b"category,value,extra\nnew,41,ignored\nnew,42,ignored\n"
                }
                at.run()
                widget(at.button, "Generate Predictions").click().run()
                self.assert_clean(at)
                self.assertEqual(len(at.session_state["prediction_results"]), 2)
                self.assertIn("Predicted target", at.session_state["prediction_results"])
                self.assertEqual(len(at.get("download_button")), 1)
                at.run()
                self.assertEqual(len(at.get("download_button")), 1)
                self.assertIs(at.session_state["ann_model"], model)
                if task == "Binary Classification":
                    widget(at.number_input, "Select Prediction Threshold").set_value(0.7).run()
                    self.assertIs(at.session_state["ann_model"], model)
                    self.assertNotIn("prediction_results", at.session_state)
                at.session_state["test_uploads"] = {
                    "Upload new data for predictions:": b"wrong_column\n1\n"
                }
                at.run()
                self.assert_clean(at)
                self.assertTrue(any("missing required" in w.value for w in at.warning))
                self.assertFalse(any(b.label == "Generate Predictions" for b in at.button))
                self.assertNotIn("prediction_results", at.session_state)

    def test_area_preserves_zeroes_and_source(self):
        df = pd.DataFrame({"x": ["a", "b", "c"], "y": [0, 1, 2]})
        original = df.copy()
        fig = fig_area_chart(df, "x", "y")
        pd.testing.assert_frame_equal(df, original)
        np.testing.assert_array_equal(fig.data[0].y, [0, 1, 2])

    def test_regression_plot_uses_actual_r2(self):
        fig = plot_predicted_vs_actual(np.array([1, 2, 3]), np.array([11, 12, 13]))
        self.assertIn("R²=-149.000", fig.layout.title.text)

    def test_gain_starts_at_zero_and_ranks_highest_predictions_first(self):
        fig = plot_cumulative_gain(np.array([1, 3]), np.array([1, 3]))
        np.testing.assert_array_equal(fig.data[0].x, [0, 0.5, 1])
        np.testing.assert_array_equal(fig.data[0].y, [0, 0.75, 1])
        with self.assertRaises(ValueError):
            plot_cumulative_gain(np.array([0, 0]), np.array([1, 2]))

    def test_confusion_matrix_with_missing_test_class(self):
        with np.errstate(divide="raise", invalid="raise"):
            fig = cm_map(np.array([[2, 1], [0, 0]]), [0, 1])
        self.assertEqual(list(fig.data[0].text[1]), ["0<br>0.0%", "0<br>0.0%"])


if __name__ == "__main__":
    unittest.main()
