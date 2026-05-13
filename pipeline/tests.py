import pandas as pd

import numpy as np
import pandas as pd


def _mad(x: pd.Series) -> float:
    """
    Median absolute deviation (unscaled).
    """
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))


def _safe_log_series(x: pd.Series, eps: float = 1e-12) -> pd.Series:
    """
    Log-transform a numeric series safely.
    Values <= 0 become NaN unless shifted by eps first.
    """
    x = pd.to_numeric(x, errors="coerce")
    x = x.where(x > 0, np.nan)
    return np.log(x + eps)


def validate_robust_bridge(
    old_df: pd.DataFrame,
    bridged_df: pd.DataFrame,
    new_unbridged_df: pd.DataFrame | None = None,
    features: list[str] | None = None,
    tol: float = 0.01,
    eps: float = 1e-12,
    quantiles: tuple[float, ...] = (0.05, 0.25, 0.5, 0.75, 0.95),
    check_quantiles: bool = True,
    check_z_preservation: bool = False,
) -> dict:
    """
    Validate whether a bridged dataframe matches an old dataframe in log space
    using robust summary statistics and (optionally) robust-z preservation.

    Parameters
    ----------
    old_df : pd.DataFrame
        Old-method data.
    bridged_df : pd.DataFrame
        New-method data after bridging.
    new_unbridged_df : pd.DataFrame | None
        New-method data before bridging. Required if check_z_preservation=True.
    features : list[str] | None
        Features to test. If None, use intersection of columns.
    tol : float
        Default near-equality tolerance.
    eps : float
        Small constant added before log.
    quantiles : tuple[float, ...]
        Quantiles to compare in log space.
    check_quantiles : bool
        Whether to test quantile alignment.
    check_z_preservation : bool
        Whether to test robust-z preservation:
            z_new = (log(new_unbridged) - median_new) / MAD_new
            z_bridged_vs_old = (log(bridged) - median_old) / MAD_old
        This is strongest when old/new/bridged rows correspond sample-wise.

    Returns
    -------
    dict with keys:
        - "summary": dict
        - "per_feature": pd.DataFrame
    """
    if features is None:
        features = sorted(set(old_df.columns).intersection(bridged_df.columns))
        if new_unbridged_df is not None:
            features = sorted(set(features).intersection(new_unbridged_df.columns))

    if check_z_preservation and new_unbridged_df is None:
        raise ValueError("new_unbridged_df must be provided when check_z_preservation=True")

    results = []

    for feat in features:
        old_log = _safe_log_series(old_df[feat], eps=eps)
        bridged_log = _safe_log_series(bridged_df[feat], eps=eps)

        old_log = old_log.dropna()
        bridged_log = bridged_log.dropna()

        row = {
            "feature": feat,
            "n_old": len(old_log),
            "n_bridged": len(bridged_log),
        }

        # Basic robust summaries
        med_old = np.median(old_log) if len(old_log) else np.nan
        med_br = np.median(bridged_log) if len(bridged_log) else np.nan
        mad_old = _mad(old_log)
        mad_br = _mad(bridged_log)

        row["median_old_log"] = med_old
        row["median_bridged_log"] = med_br
        row["median_abs_diff"] = np.abs(med_old - med_br) if pd.notna(med_old) and pd.notna(med_br) else np.nan
        row["median_pass"] = row["median_abs_diff"] <= tol if pd.notna(row["median_abs_diff"]) else False

        row["mad_old_log"] = mad_old
        row["mad_bridged_log"] = mad_br
        row["mad_abs_diff"] = np.abs(mad_old - mad_br) if pd.notna(mad_old) and pd.notna(mad_br) else np.nan
        row["mad_pass"] = row["mad_abs_diff"] <= tol if pd.notna(row["mad_abs_diff"]) else False

        # Quantile alignment
        quantile_passes = []
        if check_quantiles:
            for q in quantiles:
                q_old = old_log.quantile(q) if len(old_log) else np.nan
                q_br = bridged_log.quantile(q) if len(bridged_log) else np.nan
                q_diff = np.abs(q_old - q_br) if pd.notna(q_old) and pd.notna(q_br) else np.nan
                q_pass = q_diff <= tol if pd.notna(q_diff) else False

                q_label = str(q).replace(".", "_")
                row[f"q{q_label}_old"] = q_old
                row[f"q{q_label}_bridged"] = q_br
                row[f"q{q_label}_abs_diff"] = q_diff
                row[f"q{q_label}_pass"] = q_pass
                quantile_passes.append(q_pass)

            row["quantiles_all_pass"] = all(quantile_passes) if quantile_passes else True
        else:
            row["quantiles_all_pass"] = True

        # Robust-z preservation
        if check_z_preservation:
            new_log = _safe_log_series(new_unbridged_df[feat], eps=eps)

            tmp = pd.concat(
                {
                    "new_log": new_log,
                    "bridged_log": _safe_log_series(bridged_df[feat], eps=eps),
                },
                axis=1,
            ).dropna()

            med_new = np.median(new_log.dropna()) if len(new_log.dropna()) else np.nan
            mad_new = _mad(new_log.dropna())

            row["median_new_log"] = med_new
            row["mad_new_log"] = mad_new

            if (
                len(tmp) > 0
                and pd.notna(med_old)
                and pd.notna(mad_old)
                and pd.notna(med_new)
                and pd.notna(mad_new)
                and mad_old > 0
                and mad_new > 0
            ):
                z_new = (tmp["new_log"] - med_new) / mad_new
                z_bridged_vs_old = (tmp["bridged_log"] - med_old) / mad_old
                z_diff = np.abs(z_new - z_bridged_vs_old)

                row["z_max_abs_diff"] = z_diff.max()
                row["z_mean_abs_diff"] = z_diff.mean()
                row["z_pass"] = row["z_max_abs_diff"] <= tol
            else:
                row["z_max_abs_diff"] = np.nan
                row["z_mean_abs_diff"] = np.nan
                row["z_pass"] = False
        else:
            row["z_max_abs_diff"] = np.nan
            row["z_mean_abs_diff"] = np.nan
            row["z_pass"] = True

        row["all_pass"] = (
            row["median_pass"]
            and row["mad_pass"]
            and row["quantiles_all_pass"]
            and row["z_pass"]
        )

        results.append(row)

    per_feature = pd.DataFrame(results)
    median_failed_features = (
        per_feature.loc[~per_feature["median_pass"], "feature"].tolist()
    )

    summary = {
        "n_features_tested": len(per_feature),
        "n_all_pass": int(per_feature["all_pass"].sum()) if len(per_feature) else 0,
        "all_features_pass": bool(per_feature["all_pass"].all()) if len(per_feature) else False,
        "n_median_pass": int(per_feature["median_pass"].sum()) if len(per_feature) else 0,
        "n_mad_pass": int(per_feature["mad_pass"].sum()) if len(per_feature) else 0,
        "n_quantile_pass": int(per_feature["quantiles_all_pass"].sum()) if len(per_feature) else 0,
        "n_z_pass": int(per_feature["z_pass"].sum()) if len(per_feature) else 0,
        "median_failed_features": median_failed_features,
        "tolerance": tol,
    }

    return {
        "summary": summary,
        "per_feature": per_feature,
    }