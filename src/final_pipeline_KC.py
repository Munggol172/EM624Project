"""
EM624 Final Project – Suicide in the United States (Gun-related, 2012–2014)

This script implements a modular analysis pipeline for firearm suicides in the U.S.
using the FiveThirtyEight full_data.csv dataset (2012–2014).

Expected project structure (relative to this script):
- data/full_data.csv
- output/plots/
- output/tables/
- output/results/

Run:
    python final_pipeline.py
"""

import os
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


# -------------------------------------------------------------------
# COMPONENT 0 – DIRECTORY SETUP
# -------------------------------------------------------------------
# LLM PROMPT (Component 0 – setup_output_directories)
# "For an EM624 final project on U.S. firearm suicides, write a small, reusable
# Python utility function called setup_output_directories(base_dir: str = 'output')
# that creates three subfolders: plots, tables, and results. It should return a
# dictionary of Path objects pointing to those folders and print a short log of
# what was created. Use pathlib and os.makedirs with exist_ok=True."


def setup_output_directories(base_dir: str = "output") -> Dict[str, Path]:
    """
    Create and/or verify the output directory structure for the project.

    Parameters
    ----------
    base_dir : str, optional
        Base output directory to use. Defaults to "output".

    Returns
    -------
    dict
        Dictionary with keys 'plots', 'tables', and 'results', each mapped to
        a pathlib.Path instance.
    """
    base_path = Path(base_dir)
    plots_path = base_path / "plots"
    tables_path = base_path / "tables"
    results_path = base_path / "results"

    for p in [plots_path, tables_path, results_path]:
        os.makedirs(p, exist_ok=True)

    print(f"[setup] Output directories ready under: {base_path.resolve()}")
    return {
        "plots": plots_path,
        "tables": tables_path,
        "results": results_path,
    }


# -------------------------------------------------------------------
# COMPONENT 1 – DATA INGESTION
# -------------------------------------------------------------------
# LLM PROMPT (Component 1 – load_raw_data)
# "Using pandas, implement a function load_raw_data(data_dir: str = 'data')
# for an EM624 project on firearm suicides. It should load the FiveThirtyEight
# 'full_data.csv' file from the given directory, print how many rows and columns
# were loaded, and return a pandas DataFrame. Use defensive coding: check for
# file existence and raise a clear ValueError if the file is missing."


def load_raw_data(data_dir: str = "data") -> pd.DataFrame:
    """
    Load the main FiveThirtyEight gun deaths dataset.

    Parameters
    ----------
    data_dir : str, optional
        Directory where full_data.csv is stored. Defaults to "data".

    Returns
    -------
    pandas.DataFrame
        Raw dataset as loaded from disk.
    """
    data_path = Path(data_dir) / "full_data.csv"
    if not data_path.exists():
        raise ValueError(f"Expected dataset not found at: {data_path}")

    df = pd.read_csv(data_path)
    print(f"[load] Loaded dataset from {data_path} with shape {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# -------------------------------------------------------------------
# COMPONENT 2 – FILTER TO FIREARM SUICIDES
# -------------------------------------------------------------------
# LLM PROMPT (Component 2 – subset_firearm_suicides)
# "Write a function subset_firearm_suicides(df: pd.DataFrame) for the
# FiveThirtyEight gun deaths dataset. It should filter the rows to only those
# with intent == 'Suicide', optionally dropping records with missing age, sex,
# or race because they are critical demographic variables. The function should
# return a cleaned copy of the filtered DataFrame and print how many rows
# remain after filtering."


def subset_firearm_suicides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subset the full dataset to gun deaths classified as suicides.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw gun deaths dataset.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only suicide cases with key demographics present.
    """
    print("[filter] Subsetting to suicides based on intent == 'Suicide'...")
    mask_suicide = df["intent"] == "Suicide"
    df_suicide = df.loc[mask_suicide].copy()

    # Drop rows missing core demographics
    before_drop = df_suicide.shape[0]
    df_suicide = df_suicide.dropna(subset=["age", "sex", "race"])
    after_drop = df_suicide.shape[0]

    print(f"[filter] Suicides: {before_drop:,} rows → {after_drop:,} rows after dropping missing age/sex/race")
    return df_suicide


# -------------------------------------------------------------------
# COMPONENT 3 – FEATURE ENGINEERING
# -------------------------------------------------------------------
# LLM PROMPT (Component 3 – engineer_features)
# "For an EM624 suicide analysis project, implement engineer_features(df:
# pd.DataFrame) that: (1) coerces age to numeric and creates 10-year age
# bands, (2) maps the FiveThirtyEight education codes (1–5) to readable
# labels, (3) adds a 'season' and 'month_name' variable based on the integer
# month, and (4) returns a new DataFrame without modifying the input in
# place. Use clear labels appropriate for an academic report."


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform basic feature engineering: age groups, education labels, season, etc.

    Parameters
    ----------
    df : pandas.DataFrame
        Suicide-only dataset.

    Returns
    -------
    pandas.DataFrame
        Suicide dataset with engineered features added.
    """
    print("[features] Engineering demographic and temporal variables...")
    df_feat = df.copy()

    # Age as numeric and age groups (10-year bins)
    df_feat["age"] = pd.to_numeric(df_feat["age"], errors="coerce")

    age_bins = [0, 20, 30, 40, 50, 60, 70, 80, np.inf]
    age_labels = ["0–19", "20–29", "30–39", "40–49", "50–59", "60–69", "70–79", "80+"]
    df_feat["age_group"] = pd.cut(df_feat["age"], bins=age_bins, labels=age_labels, right=False)

    # Education mapping (FiveThirtyEight coding)
    education_map = {
        1.0: "Less than HS",
        2.0: "HS / GED",
        3.0: "Some college",
        4.0: "Bachelor's",
        5.0: "Graduate",
    }
    # If education is stored as float or int, map directly
    df_feat["education_level"] = df_feat["education"].map(education_map)

    # Month to season and month name
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    }
    df_feat["season"] = df_feat["month"].map(season_map)
    df_feat["month_name"] = pd.to_datetime(df_feat["month"], format="%m").dt.month_name()

    print("[features] Finished feature engineering (age_group, education_level, season, month_name).")
    return df_feat


# -------------------------------------------------------------------
# COMPONENT 4 – DESCRIPTIVE TABLES
# -------------------------------------------------------------------
# LLM PROMPT (Component 4 – descriptive_tables)
# "Create a function descriptive_tables(df: pd.DataFrame, tables_dir: Path)
# that builds simple descriptive statistics for a suicide dataset: frequency
# tables for age_group, sex, race, and education_level, plus a cross-tab of
# age_group by sex. Each table should be saved as a CSV in the tables_dir,
# returned in a dictionary, and accompanied by concise print statements that
# tell the user where the files were written."


def descriptive_tables(df: pd.DataFrame, tables_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Generate and save basic descriptive tables for key variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Suicide dataset with engineered features.
    tables_dir : pathlib.Path
        Directory where tables should be saved.

    Returns
    -------
    dict
        Mapping from table name to DataFrame.
    """
    print("[tables] Creating descriptive tables...")

    tables: Dict[str, pd.DataFrame] = {}

    # Simple frequency distributions
    for var in ["age_group", "sex", "race", "education_level", "season", "year"]:
        if var in df.columns:
            freq = df[var].value_counts(dropna=False).rename("count").to_frame()
            freq["percent"] = 100 * freq["count"] / freq["count"].sum()
            tables[f"{var}_distribution"] = freq
            out_path = tables_dir / f"{var}_distribution.csv"
            freq.to_csv(out_path)
            print(f"[tables] Saved {var} distribution to {out_path}")

    # Cross-tab of age group by sex
    if "age_group" in df.columns and "sex" in df.columns:
        ctab_age_sex = pd.crosstab(df["age_group"], df["sex"])
        tables["age_group_by_sex"] = ctab_age_sex
        out_path = tables_dir / "age_group_by_sex.csv"
        ctab_age_sex.to_csv(out_path)
        print(f"[tables] Saved age_group × sex table to {out_path}")

    return tables


# -------------------------------------------------------------------
# COMPONENT 5 – VISUALIZATION SUITE
# -------------------------------------------------------------------
# LLM PROMPT (Component 5 – visualization_suite)
# "Write a visualization_suite(df: pd.DataFrame, plots_dir: Path) function
# for a suicide dataset that uses seaborn/matplotlib to generate several
# static PNG plots: (a) bar chart of suicides by age_group, (b) bar chart by
# sex, (c) bar chart by race, (d) bar chart by education_level, (e) yearly
# trend from 2012–2014, and (f) a heatmap of age_group vs sex. Each figure
# should be saved into plots_dir with an informative filename and have
# publication-style titles and axis labels suitable for a graduate report."


def visualization_suite(df: pd.DataFrame, plots_dir: Path) -> None:
    """
    Create and save a set of exploratory plots for the suicide dataset.

    Parameters
    ----------
    df : pandas.DataFrame
        Suicide dataset with engineered features.
    plots_dir : pathlib.Path
        Directory where plots should be saved.
    """
    print("[plots] Generating visualizations...")
    sns.set(style="whitegrid")

    # Helper to reduce repeated code
    def bar_plot(variable: str, title: str, filename: str, order=None):
        plt.figure(figsize=(10, 6))
        if order is None:
            order_local = df[variable].value_counts().index
        else:
            order_local = order
        sns.countplot(data=df, x=variable, order=order_local)
        plt.title(title, fontsize=14)
        plt.xlabel(variable.replace("_", " ").title())
        plt.ylabel("Number of suicides")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        out_path = plots_dir / filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plots] Saved {title} to {out_path}")

    # Age, sex, race, education, season, year
    if "age_group" in df.columns:
        bar_plot("age_group", "Firearm suicides by age group", "age_group_bar.png")
    if "sex" in df.columns:
        bar_plot("sex", "Firearm suicides by sex", "sex_bar.png")
    if "race" in df.columns:
        bar_plot("race", "Firearm suicides by race", "race_bar.png")
    if "education_level" in df.columns:
        bar_plot("education_level", "Firearm suicides by education level", "education_bar.png")
    if "season" in df.columns:
        bar_plot("season", "Firearm suicides by season", "season_bar.png")
    if "year" in df.columns:
        # Ensure chronological order
        year_order = sorted(df["year"].unique())
        bar_plot("year", "Firearm suicides by year (2012–2014)", "year_bar.png", order=year_order)

    # Heatmap: age group × sex
    if "age_group" in df.columns and "sex" in df.columns:
        plt.figure(figsize=(8, 6))
        ctab = pd.crosstab(df["age_group"], df["sex"])
        sns.heatmap(ctab, annot=True, fmt="d")
        plt.title("Firearm suicides – age group × sex")
        plt.ylabel("Age group")
        plt.xlabel("Sex")
        plt.tight_layout()
        out_path = plots_dir / "age_group_by_sex_heatmap.png"
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"[plots] Saved age_group × sex heatmap to {out_path}")

    print("[plots] All requested plots generated.")


# -------------------------------------------------------------------
# COMPONENT 6 – RISK & ASSOCIATION ANALYSIS
# -------------------------------------------------------------------
# LLM PROMPT (Component 6 – risk_analysis_and_tests)
# "Implement risk_analysis_and_tests(df: pd.DataFrame) for a firearm suicide
# dataset. It should: (1) compute male-to-female burden ratios (count_male /
# count_female), (2) identify the age_group with the largest number of
# suicides overall, (3) compute a chi-square test of independence between
# age_group and sex, and (4) return a dictionary summarizing these statistics
# in plain numbers (counts, ratios, p-values). The function should print the
# key findings to the console in a clear, academic style."


def risk_analysis_and_tests(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform simple burden ratios and chi-square tests for key variables.

    Parameters
    ----------
    df : pandas.DataFrame
        Suicide dataset with engineered features.

    Returns
    -------
    dict
        Summary statistics including burden ratios and chi-square results.
    """
    print("[stats] Performing burden and association analyses...")
    summary: Dict[str, Any] = {}

    # Gender burden ratio
    if "sex" in df.columns:
        gender_counts = df["sex"].value_counts()
        n_male = gender_counts.get("M", 0)
        n_female = gender_counts.get("F", 0)
        burden_ratio = np.nan
        if n_female > 0:
            burden_ratio = n_male / n_female

        summary["gender_counts"] = gender_counts.to_dict()
        summary["male_female_burden_ratio"] = burden_ratio

        print(f"[stats] Suicides by sex – M: {n_male:,}, F: {n_female:,}")
        if not np.isnan(burden_ratio):
            print(f"[stats] Male-to-female burden ratio (counts only): {burden_ratio:.2f}")

    # Age group with highest burden
    if "age_group" in df.columns:
        age_counts = df["age_group"].value_counts()
        top_age_group = age_counts.idxmax()
        top_age_count = age_counts.max()
        summary["age_group_counts"] = age_counts.to_dict()
        summary["highest_burden_age_group"] = str(top_age_group)
        summary["highest_burden_age_group_count"] = int(top_age_count)
        print(f"[stats] Highest burden age group: {top_age_group} ({top_age_count:,} suicides)")

    # Chi-square test: age_group × sex
    if "age_group" in df.columns and "sex" in df.columns:
        contingency = pd.crosstab(df["age_group"], df["sex"])
        chi2, p_val, dof, expected = chi2_contingency(contingency)
        summary["chi2_age_sex"] = {
            "chi2": chi2,
            "p_value": p_val,
            "dof": dof,
        }
        print(f"[stats] Chi-square test (age_group × sex): χ² = {chi2:.2f}, dof = {dof}, p = {p_val:.4g}")

    print("[stats] Statistical analysis complete.")
    return summary


# -------------------------------------------------------------------
# COMPONENT 7 – NARRATIVE FINDINGS & RECOMMENDATIONS
# -------------------------------------------------------------------
# LLM PROMPT (Component 7 – generate_narrative_findings)
# "Write a function generate_narrative_findings(df: pd.DataFrame,
# stats_summary: dict, results_dir: Path) that produces a short textual
# summary (as a .txt file) of key empirical insights from a firearm suicide
# dataset. The narrative should: (a) interpret the male–female burden ratio,
# (b) describe which age_group appears most affected, (c) comment on whether
# the age_group × sex chi-square test suggests a relationship, and (d) list a
# few high-level, actionable recommendations for suicide prevention
# professionals. The function should also print the same narrative to the
# console."


def generate_narrative_findings(
    df: pd.DataFrame,
    stats_summary: Dict[str, Any],
    results_dir: Path,
    filename: str = "narrative_summary.txt",
) -> None:
    """
    Generate a narrative summary and basic recommendations based on the analysis.

    Parameters
    ----------
    df : pandas.DataFrame
        Suicide dataset with engineered features.
    stats_summary : dict
        Output from risk_analysis_and_tests.
    results_dir : pathlib.Path
        Directory where the narrative text file will be written.
    filename : str, optional
        Name of the text file to create.
    """
    print("[summary] Writing narrative findings and recommendations...")
    lines = []

    lines.append("EM624 Firearm Suicide Analysis – Key Findings\n")
    lines.append("------------------------------------------------------------\n")

    # Interpret gender burden
    gender_counts = stats_summary.get("gender_counts", {})
    ratio = stats_summary.get("male_female_burden_ratio", np.nan)
    if gender_counts:
        lines.append("Gender burden:\n")
        lines.append(f"  • Male suicides:   {gender_counts.get('M', 0):,}\n")
        lines.append(f"  • Female suicides: {gender_counts.get('F', 0):,}\n")
        if not np.isnan(ratio):
            lines.append(f"  • Within firearm suicides, there are approximately {ratio:.1f} male cases for every female case.\n")
        lines.append("\n")

    # Age group burden
    top_age = stats_summary.get("highest_burden_age_group")
    top_age_count = stats_summary.get("highest_burden_age_group_count")
    if top_age is not None:
        lines.append("Age patterns:\n")
        lines.append(f"  • The most affected age band is {top_age}, with about {top_age_count:,} firearm suicides in 2012–2014.\n")
        lines.append("  • This suggests that mid- to later-life adults deserve targeted screening and outreach.\n\n")

    # Chi-square interpretation
    chi2_info = stats_summary.get("chi2_age_sex")
    if chi2_info is not None:
        p_val = chi2_info["p_value"]
        lines.append("Age–gender relationship:\n")
        lines.append(
            f"  • A chi-square test of age_group × sex yields p = {p_val:.4g}, "
            "indicating that the age distribution of suicides differs by gender.\n\n"
        )

    # High-level recommendations
    lines.append("Preliminary recommendations (for discussion in the report):\n")
    lines.append("  • Prioritize outreach to high-burden groups (e.g., middle-aged and older men) through\n")
    lines.append("    primary care, veterans’ services, and community mental health programs.\n")
    lines.append("  • Expand culturally informed, community-based interventions in states and populations\n")
    lines.append("    with elevated firearm suicide rates.\n")
    lines.append("  • Improve data linkage between health, social services, and firearm injury registries to\n")
    lines.append("    better identify risk trajectories and intervention points.\n")
    lines.append("  • Support longitudinal research on how economic stress, mental health access, and\n")
    lines.append("    firearm availability jointly shape suicide risk.\n")

    out_path = results_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

    # Also print to console for convenience
    print("\n" + "".join(lines))
    print(f"[summary] Narrative written to {out_path}")


# -------------------------------------------------------------------
# MAIN PIPELINE ORCHESTRATOR
# -------------------------------------------------------------------
# LLM PROMPT (Component 8 – main)
# "Implement a main() function that orchestrates the entire EM624 suicide
# analysis pipeline by: (1) setting up the output directories, (2) loading the
# raw data, (3) subsetting to firearm suicides, (4) engineering features,
# (5) generating descriptive tables, (6) producing visualizations, (7) running
# risk and association analyses, and (8) creating a narrative text summary.
# The function should print a clear 'Pipeline complete' message that tells the
# user where to look for plots, tables, and results."


def main() -> None:
    """Run the full analysis pipeline end-to-end."""
    print("========== EM624 Firearm Suicide Analysis Pipeline ==========")

    # 0. Directories
    dirs = setup_output_directories(base_dir="output")

    # 1. Load data
    df_raw = load_raw_data(data_dir="data")

    # 2. Filter to suicides
    df_suicide = subset_firearm_suicides(df_raw)

    # 3. Feature engineering
    df_feat = engineer_features(df_suicide)

    # 4. Descriptive tables
    descriptive_tables(df_feat, tables_dir=dirs["tables"])

    # 5. Visualizations
    visualization_suite(df_feat, plots_dir=dirs["plots"])

    # 6. Statistical analysis
    stats_summary = risk_analysis_and_tests(df_feat)

    # 7. Narrative findings
    generate_narrative_findings(df_feat, stats_summary, results_dir=dirs["results"])

    print("\n[done] Pipeline complete.")
    print(f"       Plots:   {dirs['plots'].resolve()}")
    print(f"       Tables:  {dirs['tables'].resolve()}")
    print(f"       Results: {dirs['results'].resolve()}")
    print("============================================================")


if __name__ == "__main__":
    main()
