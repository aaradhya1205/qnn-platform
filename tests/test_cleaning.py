import numpy as np
import pandas as pd
import pytest

from qnn_platform.cleaning import (CleaningConfig, FittedCleaner, clean_structural, numeric_text_share,
                                    profile_dataframe)
from qnn_platform.demo_data import lesson_messy_table, load_demo, messy_clinical


def _messy():
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "patient_id": [f"P{i:04d}" for i in range(n)],
        "age": rng.normal(50, 10, n),
        "score": [f"{v:.0f}%" for v in rng.uniform(40, 99, n)],
        "site": rng.choice(["North", "north ", "South"], n),
        "phase": "II",
        "label": rng.choice(["a", "b"], n),
    })
    df.loc[3, "age"] = np.nan
    df.loc[7, "age"] = 5000.0
    df.loc[10, "label"] = None
    df = pd.concat([df, df.iloc[[1, 2]]], ignore_index=True)
    return df


def test_profile_finds_every_kind_of_problem():
    r = profile_dataframe(_messy(), target="label")
    assert r.duplicate_rows == 2
    assert "phase" in r.constant_cols
    assert "patient_id" in r.id_like_cols
    assert "score" in r.numeric_text_cols
    assert r.outlier_cells >= 1
    assert r.target_missing == 1
    assert r.missing_cells >= 2
    assert 0 <= r.score < 100
    assert r.issues and r.issues[0].severity in ("high", "medium", "low")


def test_clean_data_scores_100():
    df = pd.DataFrame({"a": np.arange(50.0), "b": np.arange(50.0)[::-1] * 2, "y": ["u", "v"] * 25})
    assert profile_dataframe(df, "y").score == 100


def test_numeric_text_share_handles_symbols_and_placeholders():
    s = pd.Series(["85%", "1,200", "12*", "-", "n/a", "7.5"])
    assert numeric_text_share(s) == 1.0
    assert numeric_text_share(pd.Series(["north", "south", "east"])) == 0.0


def test_structural_cleaning_applies_each_step_and_logs_it():
    df = _messy()
    clean, log = clean_structural(df, "label", CleaningConfig())
    assert "patient_id" not in clean.columns and "phase" not in clean.columns
    assert clean["score"].dtype.kind == "f"            # '85%' -> 85.0
    assert sorted(clean["site"].dropna().unique()) == ["north", "south"]   # 'North' / 'north ' merged
    assert clean.duplicated().sum() == 0
    assert clean["label"].notna().all()
    steps = [e.step for e in log]
    assert "Drop rows without a target" in steps and "Remove duplicate rows" in steps and "Convert text to numbers" in steps
    assert all(e.rows_after <= e.rows_before for e in log)


def test_each_structural_step_can_be_switched_off():
    clean, _ = clean_structural(_messy(), "label", CleaningConfig(drop_duplicates=False, drop_id_like=False,
                                                                  drop_constant=False, parse_numeric_text=False))
    assert "patient_id" in clean.columns and "phase" in clean.columns
    assert clean["score"].dtype == object
    assert clean.duplicated().sum() > 0


def test_fitted_cleaner_uses_training_statistics_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan, 1000.0]})
    cleaner = FittedCleaner(CleaningConfig(impute_numeric="median", outliers="clip", outlier_iqr=1.5)).fit(train)
    out = cleaner.transform(test)
    assert out["x"].iloc[0] == 2.0                      # train median, not influenced by the 1000 in the test set
    assert out["x"].iloc[1] < 10                        # clipped to limits learned from training rows


def test_missing_flags_and_categorical_fill():
    train = pd.DataFrame({"x": [1.0, np.nan, 3.0], "c": ["a", "a", None]})
    cleaner = FittedCleaner(CleaningConfig(add_missing_flags=True)).fit(train)
    out = cleaner.transform(train)
    assert list(out["x_was_missing"]) == [0, 1, 0]
    assert out["c"].iloc[2] == "a"


def test_demo_messy_datasets_really_are_messy():
    r = profile_dataframe(messy_clinical(), "diagnosis")
    assert r.score < 85 and r.duplicate_rows >= 10 and "patient_id" in r.id_like_cols
    assert "lab_ratio" in r.numeric_text_cols and "study_phase" in r.constant_cols
    lt = lesson_messy_table()
    rl = profile_dataframe(lt, "passed")
    assert rl.duplicate_rows == 3 and rl.outlier_cells >= 2


def test_cleaning_the_messy_demo_raises_the_score():
    demo = load_demo("messy_clinical")
    before = profile_dataframe(demo.df, demo.target)
    clean, _ = clean_structural(demo.df, demo.target, CleaningConfig())
    after = profile_dataframe(clean, demo.target)
    assert after.score > before.score and after.duplicate_rows == 0 and not after.id_like_cols
