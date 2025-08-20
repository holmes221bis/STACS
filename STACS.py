#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Trajectory Data Analysis Script

Features:
1. Processes two specified vessel trajectory data files.
2. Performs spatio-temporal clustering using ST-DBSCAN and post-processes noise.
3. Generates adaptive time windows and extracts features.
4. Trains an XGBoost classification model.
5. Predicts on test data and performs Run-Length Encoding (RLE) post-processing.
6. Maps window-level predictions back to point-level predictions.
7. Saves results and models to specified directories.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyproj
from scipy.spatial import cKDTree, KDTree
from shapely.geometry import MultiPoint
from scipy.stats import entropy
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import joblib

warnings.filterwarnings("ignore")

# -----------------------------
# Global Configuration & Path Settings
# -----------------------------
BASE_DIR = "/fish_pattern_exp"
DATA_FILES = [
    os.path.join(BASE_DIR, "data/exp_data", "danish_fishing_trajs.csv"),
    os.path.join(BASE_DIR, "data/exp_data", "gfw_trawler_trajs.csv")
]

RESULT_DIR = os.path.join(BASE_DIR, "data/result")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------------
# Global Parameters (Adjust as Needed)
# -----------------------------
DELTA_V    = 0.5   # Speed change threshold
THETA_THR  = 30    # Turning angle threshold
V_EPS      = 2.0   # Low speed threshold
RADIUS     = 500   # Spatial neighborhood radius (meters)
M          = 8     # Bearing discretization bins
MIN_PTS    = 3     # ST-DBSCAN parameter: minimum points
DIST_MULT  = 5     # Multiplier for spatial epsilon
TIME_MULT  = 5     # Multiplier for temporal epsilon

# -----------------------------
# 1. Data Loading and Preprocessing
# -----------------------------
def load_data(data_path):
    """
    Load vessel trajectory data and perform initial preprocessing.

    Parameters:
        data_path (str): Path to the data file.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = pd.read_csv(data_path)
    df['t'] = pd.to_datetime(df['timestamp'])
    df = df.rename(columns={
        'mmsi_id': 'mmsi_id',
        'lat': 'latitude',
        'lon': 'longitude',
        'speed': 'euc_speed',
        'course': 'bearing'
    })
    df.sort_values(by=['mmsi_id', 't'], inplace=True)
    df['label'] = df['label'].astype(int)
    return df

def project_latlon_to_meters(df):
    """
    Project latitude and longitude coordinates to a planar coordinate system (EPSG:3857).

    Parameters:
        df (pd.DataFrame): Original DataFrame containing latitude and longitude.

    Returns:
        pd.DataFrame: DataFrame with added 'x' and 'y' columns.
    """
    proj = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    x, y = proj.transform(df['longitude'].values, df['latitude'].values)
    df['x'] = x
    df['y'] = y
    return df

# -----------------------------
# 2. ST-DBSCAN Clustering and Noise Post-processing
# -----------------------------
def estimate_params_ship(df_ship):
    """
    Estimate the average distance and average time interval for a single vessel.

    Parameters:
        df_ship (pd.DataFrame): DataFrame for a single vessel.

    Returns:
        tuple: (avg_dist, avg_time)
    """
    if len(df_ship) < 2:
        return 100.0, 600.0
    df_ship = df_ship.sort_values('t').reset_index(drop=True)
    df_ship['prev_x'] = df_ship['x'].shift(1)
    df_ship['prev_y'] = df_ship['y'].shift(1)
    df_ship['prev_t'] = df_ship['t'].shift(1)

    valid = ~df_ship['prev_x'].isna()
    df_ship = df_ship[valid].copy()

    df_ship['dx'] = df_ship['x'] - df_ship['prev_x']
    df_ship['dy'] = df_ship['y'] - df_ship['prev_y']
    df_ship['dist'] = np.sqrt(df_ship['dx']**2 + df_ship['dy']**2)
    df_ship['delta_t'] = (df_ship['t'] - df_ship['prev_t']).dt.total_seconds()

    avg_dist = df_ship['dist'].mean() if not df_ship.empty else 100.0
    avg_time = df_ship['delta_t'].mean() if not df_ship.empty else 600.0
    return avg_dist, avg_time

def find_neighbors_kdtree_single(df_ship, eps_s, eps_t):
    """
    Find neighbors for each point using KDTree.

    Parameters:
        df_ship (pd.DataFrame): DataFrame for a single vessel.
        eps_s (float): Spatial radius.
        eps_t (float): Temporal threshold.

    Returns:
        list: List of neighbor indices for each point.
    """
    coords = df_ship[['x', 'y']].values
    times = pd.to_datetime(df_ship['t']).astype(np.int64) // 10**9  # Convert to seconds
    tree = cKDTree(coords)

    sorted_idx = np.argsort(times)
    sorted_times = times[sorted_idx]

    neighbors = []
    for idx, (xx, yy) in enumerate(coords):
        t = times[idx]
        spat_nbs = tree.query_ball_point([xx, yy], r=eps_s)
        lower_t, upper_t = t - eps_t, t + eps_t
        left = np.searchsorted(sorted_times, lower_t, side='left')
        right = np.searchsorted(sorted_times, upper_t, side='right')
        time_nbs = set(sorted_idx[left:right])
        spat_nbs_set = set(spat_nbs)
        final = list(spat_nbs_set.intersection(time_nbs))
        final = [f for f in final if f != idx]
        neighbors.append(final)
    return neighbors

def st_dbscan_single_ship(df_ship, eps_s, eps_t, min_pts):
    """
    Apply ST-DBSCAN clustering to a single vessel's data.

    Parameters:
        df_ship (pd.DataFrame): DataFrame for a single vessel.
        eps_s (float): Spatial radius.
        eps_t (float): Temporal threshold.
        min_pts (int): Minimum number of neighbors.

    Returns:
        pd.Series: Cluster labels.
    """
    n_points = len(df_ship)
    if n_points < 1:
        return pd.Series([], dtype=int, index=df_ship.index)

    nbrs = find_neighbors_kdtree_single(df_ship, eps_s, eps_t)
    cluster_assign = [-1] * n_points
    cluster_id = 0

    for i in range(n_points):
        if cluster_assign[i] != -1:
            continue
        if len(nbrs[i]) < min_pts:
            cluster_assign[i] = -1
        else:
            cluster_id += 1
            cluster_assign[i] = cluster_id
            seeds = nbrs[i].copy()
            while seeds:
                c = seeds.pop()
                if cluster_assign[c] == -1:
                    cluster_assign[c] = cluster_id
                if cluster_assign[c] != -1:
                    continue
                cluster_assign[c] = cluster_id
                if len(nbrs[c]) >= min_pts:
                    seeds.extend(nbrs[c])
    return pd.Series(cluster_assign, index=df_ship.index)

def postprocess_noise_local(df_ship, min_noise_len=3):
    """
    Merge consecutive noise points (label=-1) of length >= min_noise_len into new clusters.

    Parameters:
        df_ship (pd.DataFrame): DataFrame for a single vessel containing 'st_cluster' column.
        min_noise_len (int): Minimum length of noise to merge.

    Returns:
        pd.DataFrame: DataFrame with added 'st_cluster_pp' column.
    """
    df_ship = df_ship.sort_values('t').reset_index(drop=True)
    cvals = df_ship['st_cluster'].values
    max_label = cvals.max() if len(cvals) > 0 else -1
    new_label = max_label + 1 if max_label != -1 else 0

    i = 0
    while i < len(cvals):
        if cvals[i] == -1:
            start = i
            while i < len(cvals) and cvals[i] == -1:
                i += 1
            end = i
            if (end - start) >= min_noise_len:
                cvals[start:end] = new_label
                new_label += 1
        else:
            i += 1

    df_ship['st_cluster_pp'] = cvals
    return df_ship

# -----------------------------
# 3. Adaptive Window Generation
# -----------------------------
def generate_adaptive_windows(df_ship, desired_window_size=20, overlap_ratio=0.5):
    """
    Generate adaptive time windows for a single vessel.

    Parameters:
        df_ship (pd.DataFrame): DataFrame for a single vessel containing 'st_cluster_pp' column.
        desired_window_size (int): Desired window size (number of points).
        overlap_ratio (float): Overlap ratio between consecutive windows.

    Returns:
        list: List of window information dictionaries.
    """
    windows = []
    df_ship = df_ship.sort_values('t').reset_index(drop=True)

    cl_times = df_ship.groupby('st_cluster_pp')['t'].min().reset_index()
    cl_times = cl_times.sort_values('t').reset_index(drop=True)
    clusters = cl_times['st_cluster_pp'].values
    cluster_sizes = df_ship.groupby('st_cluster_pp').size().reindex(clusters, fill_value=0).values

    start_idx = 0
    while start_idx < len(clusters):
        total_points = 0
        end_idx = start_idx
        while end_idx < len(clusters) and total_points < desired_window_size:
            total_points += cluster_sizes[end_idx]
            end_idx += 1

        subcls = clusters[start_idx:end_idx]
        data_sub = df_ship[df_ship['st_cluster_pp'].isin(subcls)]
        if not data_sub.empty:
            windows.append({
                'start_time': data_sub['t'].iloc[0],
                'end_time'  : data_sub['t'].iloc[-1],
                'data'      : data_sub
            })

        window_size = (end_idx - start_idx + 1)
        overlap_size = int(window_size * overlap_ratio)
        next_start_idx = max_idx - overlap_size + 1
        start_idx += overlap_size

    return windows

# -----------------------------
# 4. Feature Extraction
# -----------------------------
def compute_acceleration(df, vcol='euc_speed', tcol='t'):
    """
    Compute acceleration.

    Parameters:
        df (pd.DataFrame): DataFrame window.
        vcol (str): Speed column name.
        tcol (str): Time column name.

    Returns:
        np.ndarray: Acceleration array.
    """
    dv = df[vcol].diff().fillna(0).values
    dt = df[tcol].diff().fillna(pd.Timedelta(seconds=1)).dt.total_seconds().values
    return np.where(dt > 0, dv/dt, 0)

def compute_turning_angle(df, bearing_col='bearing'):
    """
    Compute turning angle changes.

    Parameters:
        df (pd.DataFrame): DataFrame window.
        bearing_col (str): Bearing column name.

    Returns:
        np.ndarray: Turning angle change array.
    """
    dtheta = df[bearing_col].diff().fillna(0).abs().values
    dtheta = np.where(dtheta > 180, 360 - dtheta, dtheta)
    return dtheta

def extract_time_features(df):
    """
    Extract time-related features.

    Parameters:
        df (pd.DataFrame): DataFrame window.

    Returns:
        dict: Dictionary of time features.
    """
    if len(df) < 2:
        total_time = 1
    else:
        total_time = (df['t'].iloc[-1] - df['t'].iloc[0]).total_seconds()

    df = df.copy()
    df['accel'] = compute_acceleration(df, 'euc_speed', 't')
    df['dtheta'] = compute_turning_angle(df, 'bearing')

    avg_speed = df['euc_speed'].mean()
    speed_var_events = np.sum(np.abs(df['euc_speed'].diff()) > DELTA_V)
    speed_var_freq = speed_var_events / total_time if total_time > 0 else 0

    accel_mean = df['accel'].mean()
    accel_var  = df['accel'].var()

    turn_events = np.sum(df['dtheta'] > THETA_THR)
    turn_freq   = turn_events / total_time if total_time > 0 else 0

    low_speed_ratio = (df['euc_speed'] <= V_EPS).mean()

    df['low_flag'] = df['euc_speed'] <= V_EPS
    df['dwell_grp'] = (df['low_flag'] != df['low_flag'].shift()).cumsum()
    durations = df.groupby('dwell_grp')['t'].agg(['min','max']).apply(
        lambda r: (r['max'] - r['min']).total_seconds(), axis=1
    )
    is_low_grp = df.groupby('dwell_grp')['low_flag'].first()
    durations = durations[is_low_grp]
    max_dwell_time = durations.max() if not durations.empty else 0

    feats = {
        'avg_speed': avg_speed,
        'speed_var_freq': speed_var_freq,
        'accel_mean': accel_mean,
        'accel_var': accel_var,
        'turn_freq': turn_freq,
        'low_speed_ratio': low_speed_ratio,
        'max_dwell_time': max_dwell_time
    }
    return feats

def compute_trajectory_length(df):
    """
    Compute the total length of the trajectory.

    Parameters:
        df (pd.DataFrame): DataFrame window.

    Returns:
        float: Total trajectory length.
    """
    dx = df['x'].diff().fillna(0).values
    dy = df['y'].diff().fillna(0).values
    dist = np.sqrt(dx**2 + dy**2)
    return dist.sum()

def extract_spatial_features(df):
    """
    Extract spatial-related features.

    Parameters:
        df (pd.DataFrame): DataFrame window.

    Returns:
        dict: Dictionary of spatial features.
    """
    total_len = compute_trajectory_length(df)
    if len(df) < 2:
        return {
            'trajectory_length': total_len,
            'straightness_ratio': 0,
            'sinuosity': 0,
            'directional_diversity': 0,
            'convex_hull_area': 0,
            'convex_hull_ratio': 0,
            'spatial_density_var': 0
        }

    x1, y1 = df.iloc[0][['x', 'y']]
    x2, y2 = df.iloc[-1][['x', 'y']]
    dist_line = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    straightness_ratio = total_len / dist_line if dist_line > 0 else 0
    sinuosity = straightness_ratio

    theta = df['bearing'].values
    theta_bins = np.linspace(0, 360, M+1)
    tdg = np.digitize(theta, theta_bins) - 1
    tdg = np.where(tdg == M, M-1, tdg)
    counts = np.bincount(tdg, minlength=M)
    direction_div = 0
    if counts.sum() > 0:
        direction_div = entropy(counts / counts.sum(), base=np.e)

    points = df[['x', 'y']].values
    hull_area = 0
    hull_ratio = 0
    if len(points) >= 3:
        hull = MultiPoint(points).convex_hull
        hull_area = hull.area
        minx, miny, maxx, maxy = hull.bounds
        w = maxx - minx
        h = maxy - miny
        hull_ratio = w / h if h != 0 else 0

    tree = KDTree(points)
    neighbors = tree.query_ball_point(points, r=RADIUS)
    dens = np.array([len(n) for n in neighbors])
    density_var = dens.var() if len(dens) > 0 else 0

    feats = {
        'trajectory_length': total_len,
        'straightness_ratio': straightness_ratio,
        'sinuosity': sinuosity,
        'directional_diversity': direction_div,
        'convex_hull_area': hull_area,
        'convex_hull_ratio': hull_ratio,
        'spatial_density_var': density_var
    }
    return feats

def extract_features_for_window(df_window):
    """
    Extract all features for a single window.

    Parameters:
        df_window (pd.DataFrame): DataFrame window.

    Returns:
        dict: Merged dictionary of time and spatial features.
    """
    time_feats = extract_time_features(df_window)
    space_feats = extract_spatial_features(df_window)
    all_feats = {**time_feats, **space_feats}
    return all_feats

# -----------------------------
# 5. Training Phase
# -----------------------------
def segment_by_label_runs_global(df):
    """
    Segment each vessel by consecutive unchanged labels.

    Parameters:
        df (pd.DataFrame): Entire DataFrame.

    Returns:
        list: List of segmented DataFrames.
    """
    segments = []
    for mmsi, grp in df.groupby('mmsi_id', sort=False):
        grp = grp.sort_values('t').reset_index(drop=True)
        if len(grp) < 1:
            continue
        current_label = grp.loc[0, 'label']
        start = 0
        for i in range(1, len(grp)):
            lbl = grp.loc[i, 'label']
            if lbl != current_label:
                seg = grp.iloc[start:i]
                segments.append(seg)
                start = i
                current_label = lbl
        seg = grp.iloc[start:]
        segments.append(seg)
    return segments

def build_training_data_for_dataset(df, 
                                    id_col='mmsi_id',
                                    time_col='timestamp', 
                                    label_col='label',
                                    min_points=300,
                                    time_threshold=3600,
                                    overlap_ratio=5/6):
    """
    - Perform WBS on each trajectory in df (grouped by id_col).
    - Extract window features and determine window labels based on majority voting.
    - Combine all into X, y, and feature_names.

    Parameters:
        df (pd.DataFrame): Entire DataFrame.
        id_col (str): Column name for vessel ID.
        time_col (str): Column name for timestamp.
        label_col (str): Column name for labels.
        min_points (int): Minimum number of points in a window.
        time_threshold (int): Time threshold in seconds for window duration.
        overlap_ratio (float): Overlap ratio between consecutive windows.

    Returns:
        pd.DataFrame: Extracted feature DataFrame.
    """
    label_segments = segment_by_label_runs_global(df)
    feat_list = []
    for segdata in tqdm(label_segments, desc="TrainSegments"):
        if len(segdata) < 2:
            continue
        seg_label = segdata['label'].iloc[0]

        avg_dist, avg_time = estimate_params_ship(segdata)
        eps_s = avg_dist * DIST_MULT
        eps_t = avg_time * TIME_MULT

        segdata = segdata.reset_index(drop=True)
        st_labels = st_dbscan_single_ship(segdata, eps_s, eps_t, MIN_PTS)
        segdata['st_cluster'] = st_labels

        processed = postprocess_noise_local(segdata)

        # Adaptive windowing
        sub_windows = generate_adaptive_windows(processed, desired_window_size=50, overlap_ratio=0.5)
        for w in sub_windows:
            wdata = w['data']
            if len(wdata) < 2:
                continue
            feats = extract_features_for_window(wdata)
            feats['label'] = seg_label
            feats['mmsi_id'] = wdata['mmsi_id'].iloc[0]
            feats['start_time'] = wdata['t'].iloc[0]
            feats['end_time']   = wdata['t'].iloc[-1]
            feat_list.append(feats)

    train_df = pd.DataFrame(feat_list)
    return train_df

def train_xgb(train_df):
    """
    Train an XGBoost model and evaluate its performance.

    Parameters:
        train_df (pd.DataFrame): Training feature DataFrame.

    Returns:
        tuple: (Trained XGBoost model, Scaler)
    """
    X = train_df.drop(columns=['mmsi_id', 'label', 'start_time', 'end_time'], errors='ignore')
    X = X.fillna(0)
    y = train_df['label']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest  = xgb.DMatrix(X_test,  label=y_test)

    label_counts = y.value_counts()
    scale_pos_weight = 1
    if 0 in label_counts and 1 in label_counts and label_counts[1] != 0:
        scale_pos_weight = label_counts[0] / label_counts[1]

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'scale_pos_weight': scale_pos_weight,
        'seed': 42
    }

    evals = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, 100, evals, early_stopping_rounds=10, verbose_eval=False)

    y_pred_prob = bst.predict(dtest)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\n=== XGBoost Evaluation (Based on ST-DBSCAN + Adaptive Window Training) ===")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n")

    return bst, scaler

# -----------------------------
# 6. Prediction Phase (RLE)
# -----------------------------
def st_dbscan_whole_ship(df_ship):
    """
    Apply ST-DBSCAN clustering and post-process noise for a single vessel.

    Parameters:
        df_ship (pd.DataFrame): DataFrame for a single vessel.

    Returns:
        pd.DataFrame: Processed DataFrame with 'st_cluster_pp' column.
    """
    df_ship = df_ship.reset_index(drop=True)  # Reset index to match labels
    avg_dist, avg_time = estimate_params_ship(df_ship)
    eps_s = avg_dist * DIST_MULT
    eps_t = avg_time * TIME_MULT

    labels = st_dbscan_single_ship(df_ship, eps_s, eps_t, MIN_PTS)
    df_ship['st_cluster'] = labels.values  # Use .values to ensure index alignment

    processed = postprocess_noise_local(df_ship)
    return processed

def generate_adaptive_windows_global(df, desired_window_size=20, overlap_ratio=0.5):
    """
    Generate adaptive windows for the entire dataset (multiple vessels).

    Parameters:
        df (pd.DataFrame): Entire processed DataFrame containing 'st_cluster_pp' column.
        desired_window_size (int): Desired window size (number of points).
        overlap_ratio (float): Overlap ratio between consecutive windows.

    Returns:
        list: List of window information dictionaries.
    """
    windows = []
    for mmsi, grp in df.groupby('mmsi_id', sort=False):
        subwins = generate_adaptive_windows(grp, desired_window_size, overlap_ratio)
        for w in subwins:
            w['mmsi_id'] = mmsi
            windows.append(w)
    return windows

def extract_features_for_windows(windows):
    """
    Extract features for all windows.

    Parameters:
        windows (list): List of window information dictionaries.

    Returns:
        pd.DataFrame: Feature DataFrame.
    """
    feats_list = []
    for w in windows:
        d = w['data']
        if len(d) < 2:
            continue
        feats = extract_features_for_window(d)
        feats['mmsi_id'] = w['mmsi_id']
        feats['start_time'] = w['start_time']
        feats['end_time']   = w['end_time']
        feats_list.append(feats)
    return pd.DataFrame(feats_list)

def predict_with_xgb(features_df, bst, scaler):
    """
    Make predictions using the trained XGBoost model.

    Parameters:
        features_df (pd.DataFrame): Feature DataFrame.
        bst (xgb.Booster): Trained XGBoost model.
        scaler (StandardScaler): Scaler used during training.

    Returns:
        pd.DataFrame: DataFrame with added 'label_pred' and 'pred_prob' columns.
    """
    X = features_df.drop(columns=['mmsi_id', 'start_time', 'end_time'], errors='ignore')
    X = X.fillna(0)
    X_scaled = scaler.transform(X)
    dtest = xgb.DMatrix(X_scaled)
    y_prob = bst.predict(dtest)
    y_pred = (y_prob > 0.5).astype(int)
    features_df['label_pred'] = y_pred
    features_df['pred_prob']  = y_prob
    return features_df

# -----------------------------
# 7. RLE Post-processing
# -----------------------------
def run_length_encoding(labels):
    """
    Perform Run-Length Encoding (RLE).

    Parameters:
        labels (list): Sequence of labels.

    Returns:
        list: RLE result containing tuples of (label, length).
    """
    if not labels:
        return []
    rle = []
    prev = labels[0]
    count = 1
    for x in labels[1:]:
        if x == prev:
            count += 1
        else:
            rle.append((prev, count))
            prev = x
            count = 1
    rle.append((prev, count))
    return rle

def rle_to_labels(rle):
    """
    Convert RLE result back to label sequence.

    Parameters:
        rle (list): RLE result containing tuples of (label, length).

    Returns:
        list: Reconstructed label sequence.
    """
    seq = []
    for (lbl, val) in rle:
        seq.extend([lbl] * val)
    return seq

def merge_three_segment(rle):
    """
    Perform one round of three-segment pattern merging (1->0->1 or 0->1->0).

    Parameters:
        rle (list): RLE result.

    Returns:
        list: Merged RLE result.
    """
    merged = []
    i = 0
    while i < len(rle):
        lblA, lenA = rle[i]
        if i + 2 < len(rle):
            lblB, lenB = rle[i+1]
            lblC, lenC = rle[i+2]
            # 1->0->1
            if lblA == 1 and lblB == 0 and lblC == 1:
                if (lenB < lenA and lenB < lenC):
                    merged.append((1, lenA + lenB + lenC))
                    i += 3
                    continue
                else:
                    merged.append((lblA, lenA))
            # 0->1->0
            elif lblA == 0 and lblB == 1 and lblC == 0:
                if (lenB < lenA and lenB < lenC):
                    merged.append((0, lenA + lenB + lenC))
                    i += 3
                    continue
                else:
                    merged.append((lblA, lenA))
            else:
                merged.append((lblA, lenA))
        else:
            merged.append((lblA, lenA))
        i += 1
    return merged

def rle_postprocess_once(window_pred_df):
    """
    Perform RLE post-processing on window-level prediction results for each vessel.

    Parameters:
        window_pred_df (pd.DataFrame): Window-level prediction results DataFrame containing 'mmsi_id' and 'label_pred' columns.

    Returns:
        pd.DataFrame: DataFrame containing fishing segment information.
    """
    all_segments = []
    for mmsi, grp in window_pred_df.groupby('mmsi_id', sort=False):
        grp_sorted = grp.sort_values('start_time').reset_index(drop=True)
        labels = grp_sorted['label_pred'].tolist()

        # First round of three-segment merging
        rle_1 = run_length_encoding(labels)
        merged_1 = merge_three_segment(rle_1)
        final_seq = rle_to_labels(merged_1)

        # Parse fishing segments (label=1)
        idx_in_grp = 0
        seg_id = 1
        final_rle = run_length_encoding(final_seq)
        for (lbl, length) in final_rle:
            start_idx = idx_in_grp
            end_idx   = idx_in_grp + length - 1
            idx_in_grp += length

            if lbl == 1:
                seg_info = {
                    'mmsi_id': mmsi,
                    'segment_id': seg_id,
                    'start_time': grp_sorted.loc[start_idx, 'start_time'],
                    'end_time':   grp_sorted.loc[end_idx, 'end_time'],
                    'window_count': length
                }
                all_segments.append(seg_info)
                seg_id += 1

    return pd.DataFrame(all_segments)

# -----------------------------
# 8. Map Back to Point-level Predictions
# -----------------------------
def map_prediction_to_points(df_points, fishing_seg_df):
    """
    Map fishing segments back to point-level data, marking predicted fishing points.

    Parameters:
        df_points (pd.DataFrame): Entire processed DataFrame.
        fishing_seg_df (pd.DataFrame): Fishing segments DataFrame.

    Returns:
        pd.DataFrame: DataFrame with added 'predict' column.
    """
    df_out = df_points.copy()
    df_out['predict'] = 0

    for _, row in fishing_seg_df.iterrows():
        mmsi_id = row['mmsi_id']
        stt     = row['start_time']
        edt     = row['end_time']
        mask = (df_out['mmsi_id'] == mmsi_id) & (df_out['t'] >= stt) & (df_out['t'] <= edt)
        df_out.loc[mask, 'predict'] = 1

    return df_out

# -----------------------------
# 9. Main Workflow
# -----------------------------
def process_dataset(data_path):
    """
    Process a single dataset, including loading, clustering, feature extraction, model training, prediction, and result saving.

    Parameters:
        data_path (str): Path to the data file.
    """
    dataset_name = os.path.splitext(os.path.basename(data_path))[0]
    print(f"\n=== Processing Dataset: {dataset_name} ===")

    # 1. Load and Preprocess
    df = load_data(data_path)
    df = project_latlon_to_meters(df)

    # 2. Training Phase
    print("=== Training Phase ===")
    train_df = build_training_data_for_dataset(
        df,
        desired_window_size=50,
        overlap_ratio=0.5
    )
    print(f"Training feature extraction complete.")

    # 3. Train Model
    bst, scaler = train_xgb(train_df)
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_xgb_model.json")
    scaler_path = os.path.join(MODEL_DIR, f"{dataset_name}_xgb_scaler.pkl")
    bst.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"[Saved] Model => {model_path}, Scaler => {scaler_path}")

    # 4. Prediction Phase
    print("=== Prediction Phase ===")
    # 4.1 Clustering
    processed_list = []
    for mmsi, grp in tqdm(df.groupby('mmsi_id', sort=False), desc="PredictClustering"):
        df_ship = st_dbscan_whole_ship(grp.copy())
        processed_list.append(df_ship)
    df_clustered = pd.concat(processed_list, ignore_index=True)
    print("[Completed] ST-DBSCAN Clustering")

    # 4.2 Generate Windows and Extract Features
    test_windows = generate_adaptive_windows_global(
        df_clustered,
        desired_window_size=50,
        overlap_ratio=0.5
    )
    test_feat_df = extract_features_for_windows(test_windows)
    print(f"Window feature extraction complete.")

    # 4.3 Predict
    test_pred_df = predict_with_xgb(test_feat_df.copy(), bst, scaler)
    print(f"Window prediction complete.")

    # 4.4 RLE Post-processing
    fishing_seg_df = rle_postprocess_once(test_pred_df)
    print(f"Fishing segments post-processing complete.")

    # 4.5 Map Back to Point-level
    df_final = map_prediction_to_points(df_clustered, fishing_seg_df)
    out_cols = ['mmsi_id', 't', 'latitude', 'longitude', 'label', 'predict']
    df_final = df_final[out_cols].copy()
    point_pred_path = os.path.join(RESULT_DIR, f"{dataset_name}_proposed_method.csv")
    df_final.to_csv(point_pred_path, index=False)
    print(f"Point-level prediction complete.")

    print(f"\n=== Dataset {dataset_name} Processing Complete ===\n")

def main():
    """
    Main function to process all specified datasets.
    """
    for data_path in DATA_FILES:
        process_dataset(data_path)
    print("\n=== All Datasets Processed ===")

if __name__ == "__main__":
    main()
