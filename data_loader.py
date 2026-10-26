from pathlib import Path
from typing import Optional

import pandas as pd


# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

BATCH_DATA_FILE = DATA_DIR / "Program Batch Data.csv"


# ============================================================
# EXPECTED COLUMNS
# ============================================================

EXPECTED_COLUMNS = [
    "program_code",
    "tech_management",
    "previous_batch_code",
    "sales_start_date",
    "category",
    "financial_year",
    "program_status_f",
    "total_program_fees",
    "total_program_fees_sf",
    "source_created",
    "New_Repeat",
    "leads",
    "warm_leads",
    "applications",
    "enrollments",
    "gsv_cr",
    "spends",
    "Planned_Budget",
    "Planned_Lead",
    "Planned_Application",
    "Planned_Enrollment",
    "Planned_GSV",
    "paid_leads",
    "paid_enrollments",
    "paid_gsv_cr",
    "paid_spends",
    "planned_paid_leads",
    "planned_paid_enrollments",
    "planned_paid_gsv",
    "planned_paid_spends",
    "Planned_Data_Status",
    "previous_batch_leads",
    "previous_batch_warm_leads",
    "previous_batch_applications",
    "previous_batch_enrollments",
    "previous_batch_gsv_cr",
    "previous_batch_spends",
    "previous_batch_Planned_Budget",
    "previous_batch_Planned_Lead",
    "previous_batch_Planned_Application",
    "previous_batch_Planned_Enrollment",
    "previous_batch_Planned_GSV",
    "previous_batch_paid_leads",
    "previous_batch_paid_enrollments",
    "previous_batch_paid_gsv_cr",
    "previous_batch_paid_spends",
    "previous_batch_planned_paid_leads",
    "previous_batch_planned_paid_enrollments",
    "previous_batch_planned_paid_gsv",
    "previous_batch_planned_paid_spends",
]


# ============================================================
# DATA TYPES
# ============================================================

NUMERIC_COLUMNS = [
    "total_program_fees",
    "leads",
    "warm_leads",
    "applications",
    "enrollments",
    "gsv_cr",
    "spends",
    "Planned_Budget",
    "Planned_Lead",
    "Planned_Application",
    "Planned_Enrollment",
    "Planned_GSV",
    "paid_leads",
    "paid_enrollments",
    "paid_gsv_cr",
    "paid_spends",
    "planned_paid_leads",
    "planned_paid_enrollments",
    "planned_paid_gsv",
    "planned_paid_spends",
    "previous_batch_leads",
    "previous_batch_warm_leads",
    "previous_batch_applications",
    "previous_batch_enrollments",
    "previous_batch_gsv_cr",
    "previous_batch_spends",
    "previous_batch_Planned_Budget",
    "previous_batch_Planned_Lead",
    "previous_batch_Planned_Application",
    "previous_batch_Planned_Enrollment",
    "previous_batch_Planned_GSV",
    "previous_batch_paid_leads",
    "previous_batch_paid_enrollments",
    "previous_batch_paid_gsv_cr",
    "previous_batch_paid_spends",
    "previous_batch_planned_paid_leads",
    "previous_batch_planned_paid_enrollments",
    "previous_batch_planned_paid_gsv",
    "previous_batch_planned_paid_spends",
]


DATE_COLUMNS = [
    "sales_start_date",
]


# ============================================================
# CLEANING
# ============================================================

def clean_batch_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize Batch_wise_program_data.

    Does not perform business calculations.
    """

    df = df.copy()

    # Standardize column names
    df.columns = [str(col).strip() for col in df.columns]

    # Normalize Version 2 source aliases when present. Actual GSV is intentionally
    # not inferred from Planned_GSV.
    if "total_program_fees" not in df.columns and "total_program_fees_sf" in df.columns:
        df["total_program_fees"] = df["total_program_fees_sf"]

    # Convert numeric columns
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    # Convert dates
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
            )

    # Clean string columns
    string_columns = [
        "program_code",
        "tech_management",
        "previous_batch_code",
        "category",
        "financial_year",
        "program_status_f",
        "source_created",
        "New_Repeat",
        "Planned_Data_Status",
    ]

    for column in string_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .astype("string")
                .str.strip()
            )

    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_batch_data(
    df: pd.DataFrame,
    strict: bool = False,
) -> dict:
    """
    Validate whether the expected columns exist.

    strict=False:
        Returns validation information without raising.

    strict=True:
        Raises ValueError when required columns are missing.
    """

    actual_columns = set(df.columns)

    missing_columns = [
        column
        for column in EXPECTED_COLUMNS
        if column not in actual_columns
    ]

    extra_columns = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    validation = {
        "valid": len(missing_columns) == 0,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "row_count": len(df),
        "column_count": len(df.columns),
    }

    if strict and missing_columns:
        raise ValueError(
            "Batch_wise_program_data.csv is missing required columns: "
            + ", ".join(missing_columns)
        )

    return validation


# ============================================================
# LOAD
# ============================================================

def load_batch_data(
    path: Optional[str | Path] = None,
    strict_validation: bool = False,
) -> pd.DataFrame:
    """
    Load Batch_wise_program_data.csv.

    This is now the single overall analysis data source.
    """

    file_path = Path(path) if path else BATCH_DATA_FILE

    if not file_path.exists():
        raise FileNotFoundError(
            f"Batch data file not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    df = clean_batch_data(df)

    validate_batch_data(
        df,
        strict=strict_validation,
    )

    return df


# ============================================================
# SOURCE PREVIEW
# ============================================================

def get_data_preview(
    df: pd.DataFrame,
    rows: int = 20,
) -> pd.DataFrame:
    """
    Return a preview for the UI.
    """

    return df.head(rows).copy()


# ============================================================
# DATA PROFILE
# ============================================================

def get_data_profile(df: pd.DataFrame) -> dict:
    """
    Return basic information about the loaded data.
    """

    validation = validate_batch_data(df)

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "programs": (
            df["program_code"].nunique()
            if "program_code" in df.columns
            else 0
        ),
        "categories": (
            df["category"].nunique()
            if "category" in df.columns
            else 0
        ),
        "financial_years": (
            df["financial_year"].nunique()
            if "financial_year" in df.columns
            else 0
        ),
        "missing_required_columns": validation[
            "missing_columns"
        ],
    }
