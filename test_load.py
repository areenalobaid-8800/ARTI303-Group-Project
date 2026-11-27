
"""Tests for Step 1 - Load."""

import logging

import pandas as pd
import pytest

from Load import load_data


def test_load_valid_csv(tmp_path):
    """Test loading a valid CSV file."""

    file_path = tmp_path / "valid.csv"

    file_path.write_text(
        "Name,Age,Score\n"
        "Ali,20,90\n"
        "Sara,21,85\n",
        encoding="utf-8",
    )

    df = load_data(file_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 3)
    assert list(df.columns) == ["Name", "Age", "Score"]
    assert df.iloc[0]["Name"] == "Ali"
    assert df.iloc[1]["Score"] == 85


def test_load_missing_file(tmp_path):
    """Test that a missing file raises FileNotFoundError."""

    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(file_path)


def test_load_empty_file(tmp_path):
    """Test that an empty CSV file raises ValueError."""

    file_path = tmp_path / "empty.csv"
    file_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="Dataset file is empty"):
        load_data(file_path)


def test_load_csv_without_rows(tmp_path):
    """Test that a CSV containing only column names raises ValueError."""

    file_path = tmp_path / "no_rows.csv"
    file_path.write_text(
        "Name,Age,Score\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Dataset has no rows"):
        load_data(file_path)


def test_load_malformed_csv(tmp_path):
    """Test that a malformed CSV file raises ValueError."""

    file_path = tmp_path / "malformed.csv"

    file_path.write_text(
        "Name,Age\n"
        "Ali,20\n"
        "Sara,21,90\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="File is not a valid CSV"):
        load_data(file_path)


def test_load_unsupported_encoding(tmp_path):
    """Test that an unsupported file encoding raises ValueError."""

    file_path = tmp_path / "invalid_encoding.csv"

    file_path.write_bytes(
        b"Name,Age\n"
        b"Ali,\xff\xfe\n"
    )

    with pytest.raises(ValueError, match="Unsupported file encoding"):
        load_data(file_path)


def test_load_logs_success(tmp_path, caplog):
    """Test that successful loading generates an INFO log."""

    file_path = tmp_path / "logged.csv"

    file_path.write_text(
        "Name,Age\n"
        "Ali,20\n"
        "Sara,21\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.INFO, logger="Load"):
        df = load_data(file_path)

    assert "Loaded 2 rows and 2 columns" in caplog.text
    assert len(df) == 2
