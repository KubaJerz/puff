import json
import numpy as np
import os
import torch
import toml
import pandas as pd
import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from dotenv import load_dotenv
from tqdm import tqdm
from sklearn.model_selection import train_test_split

load_dotenv()

# Database configuration
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE')
}

@contextmanager
def db_connection():
    """Context manager for database connections with proper cleanup."""
    conn = None
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        yield conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        raise
    finally:
        if conn and conn.is_connected():
            conn.close()

def get_db_connection():
    """Get database connection (legacy function for compatibility)."""
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def get_project_id(project_name):
    """Get the project ID from the database."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT project_id FROM projects WHERE project_name = %s"
        cursor.execute(query, (project_name,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Project '{project_name}' not found in the database.")

def get_raw_dataset_path(project_name):
    """Get the raw dataset path for a project."""
    with db_connection() as conn:
        cursor = conn.cursor()
        
        # Get project_id
        project_id = get_project_id(project_name)
        
        # Get dataset_id from project_dataset_refs
        query = "SELECT dataset_id FROM project_dataset_refs WHERE project_id = %s"
        cursor.execute(query, (project_id,))
        ref_row = cursor.fetchone()
        
        if not ref_row:
            raise ValueError(f"No raw dataset reference found for project '{project_name}'")
        
        dataset_id = ref_row[0]
        
        # Get file_path from raw_datasets table
        query = "SELECT file_path FROM raw_datasets WHERE dataset_id = %s"
        cursor.execute(query, (dataset_id,))
        path_row = cursor.fetchone()
    
    if path_row and path_row[0]:
        return path_row[0]
    else:
        raise ValueError(f"No file_path found for dataset {dataset_id}")

def get_sessions_for_project(project_name):
    """Get all sessions for a project."""
    with db_connection() as conn:
        cursor = conn.cursor(dictionary=True)
        project_id = get_project_id(project_name)
        query = "SELECT * FROM sessions WHERE project_id = %s"
        cursor.execute(query, (project_id,))
        rows = cursor.fetchall()
    
    for row in rows:
        if 'bouts' in row and row['bouts']:
            try:
                row['bouts'] = json.loads(row['bouts'])
            except json.JSONDecodeError:
                row['bouts'] = []
    
    return rows

def get_participant_id(participant_code):
    """Get the participant ID from the database based on participant code."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT participant_id FROM participants WHERE participant_code = %s"
        cursor.execute(query, (participant_code,))
        row = cursor.fetchone()

    if row:
        return row[0]
    else:
        raise ValueError(f"Participant code '{participant_code}' not found in the database.")

def get_participant_projects(participant_id):
    """Get all projects associated with a participant."""
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT project_name FROM projects WHERE participant_id = %s"
        cursor.execute(query, (participant_id,))
        rows = cursor.fetchall()
    return [row[0] for row in rows]

def load_data(raw_dataset_path, session_name, start_ns=None, stop_ns=None, use_gyro=False):
    """Load accelerometer data for a session with optional time filtering."""
    try:
        session_path = os.path.join(raw_dataset_path, session_name.split('.')[0])
        accelerometer_path = os.path.join(session_path, 'accelerometer_data.csv')
        
        df_accel = pd.read_csv(accelerometer_path)

        if 'accel_x' not in df_accel.columns:
            df_accel.rename(columns={'x': 'accel_x', 'y': 'accel_y', 'z': 'accel_z'}, inplace=True)
        else:
            df_accel.rename(columns={'x_accel':'accel_x', 'y_accel':'accel_y', 'z_accel':'accel_y'}, inplace=True)

        
        if use_gyro:
            gyroscope_path = os.path.join(session_path, 'gyroscope_data.csv')
            df_gyro = pd.read_csv(gyroscope_path)

            if 'gyro_x' not in df_gyro.columns:
                df_gyro.rename(columns={'x': 'gyro_x', 'y': 'gyro_y', 'z': 'gyro_z'}, inplace=True)
            else:
                df_gyro.rename(columns={'x_gyro':'gyro_x', 'y_gyro':'gyro_y', 'z_gyro':'gyro_z'}, inplace=True)


        if use_gyro:
            df = pd.merge(df_accel, df_gyro, on='ns_since_reboot', how='inner')


        
        # Apply time filtering if provided
        if start_ns is not None and stop_ns is not None:
            mask = (df['ns_since_reboot'] >= start_ns) & (df['ns_since_reboot'] <= stop_ns)
            df = df[mask].copy()
            df.reset_index(drop=True, inplace=True)
        
        return df
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading data for session {session_name}: {e}")
        return pd.DataFrame()

def combine_sensor_data(session, project_path, use_gyro=False):
    """Legacy function name - calls load_data for compatibility."""
    df = load_data(project_path, session['session_name'], start_ns=session['start_ns'], stop_ns=session['stop_ns'], use_gyro=use_gyro)
    
    if df.empty:
        return df
    
    # Ensure data types
    for col in ['ns_since_reboot', 'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']:
        if col in df.columns:
            df[col] = df[col].astype(float)
    
    return df.dropna()

def validate_splits(train_percent, dev_percent, test_percent=0):
    """Validate that dataset splits add up to 1.0."""
    total = train_percent + dev_percent + test_percent
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Dataset splits must add up to 1.0, got {total}")

def check_for_gaps(df, threshold_minutes=5):
    """Split dataframe on time gaps larger than threshold."""
    if df.empty:
        return [df]
        
    gap_threshold_ns = threshold_minutes * 60 * 1_000_000_000
    df = df.sort_values('ns_since_reboot').reset_index(drop=True)
    time_diffs = df['ns_since_reboot'].diff()
    gap_indices = time_diffs[time_diffs > gap_threshold_ns].index
    
    if len(gap_indices) == 0:
        return [df]
    
    segments = []
    start_idx = 0
    
    for gap_idx in gap_indices:
        if start_idx < gap_idx:
            segment = df.iloc[start_idx:gap_idx].copy()
            if not segment.empty:
                segments.append(segment)
        start_idx = gap_idx
    
    # Add final segment
    if start_idx < len(df):
        final_segment = df.iloc[start_idx:].copy()
        if not final_segment.empty:
            segments.append(final_segment)
    
    return segments

def rename_df(df, sensor_type):
    """Rename DataFrame columns to standardized format."""
    if set(['ns_since_reboot', 'x', 'y', 'z']).issubset(set(df.columns)):
        df = df.rename(columns={"x": f"x_{sensor_type}", "y": f"y_{sensor_type}", "z": f"z_{sensor_type}"})
    elif set(['ns_since_reboot', f'{sensor_type}_x', f'{sensor_type}_y', f'{sensor_type}_z']).issubset(set(df.columns)):
        df = df.rename(columns={f"{sensor_type}_x": f"x_{sensor_type}", f"{sensor_type}_y": f"y_{sensor_type}", f"{sensor_type}_z": f"z_{sensor_type}"})
    else:   
        raise ValueError(f"Unexpected column names: {df.columns}. Expected: ['ns_since_reboot', 'x', 'y', 'z']")
    
    return df

def apply_labels_to_df(df, session, target_labels, label_value=1):
    """Add labels to dataframe based on bout annotations."""
    if df.empty:
        return df
    
    df['label'] = 0
    
    for bout in session.get('bouts', []):
        if bout.get('label') in target_labels:
            start_time = bout.get('start_time', bout.get('start'))
            end_time = bout.get('end_time', bout.get('end'))
            if start_time is not None and end_time is not None:
                mask = (df['ns_since_reboot'] >= start_time) & (df['ns_since_reboot'] <= end_time)
                df.loc[mask, 'label'] = label_value
    
    return df

def create_windows(df, window_size, step_size, use_gyro=False):
    """Create sliding windows from dataframe."""
    if len(df) < window_size:
        print(f"Warning: DataFrame too small ({len(df)} < {window_size}), skipping")
        return np.array([]), np.array([])
    
    if use_gyro:
        feature_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
    else:
        feature_cols = ['accel_x', 'accel_y', 'accel_z']
        

    X_data = df[feature_cols].values
    y_data = df['label'].values
    
    windows_X = []
    windows_y = []
    
    for i in range(0, len(df) - window_size + 1, step_size):
        window_X = X_data[i:i + window_size]
        window_y = y_data[i:i + window_size]
        windows_X.append(window_X)
        windows_y.append(window_y)
    
    return np.array(windows_X), np.array(windows_y)

def filter_negative_windows(X, y, percent_to_sample=1.0, random_seed=70):
    """Sample negative windows based on configured percentage."""
    if percent_to_sample >= 1.0:
        return X, y
    
    has_label = np.any(y > 0, axis=1)
    positive_indices = np.where(has_label)[0]
    negative_indices = np.where(~has_label)[0]
    
    print(f'Positive samples: {len(positive_indices)} : Negative Samples: {len(negative_indices)}')
    
    # Sample negative windows
    num_negative_to_keep = int(len(negative_indices) * percent_to_sample)
    np.random.seed(random_seed)
    sampled_negative_indices = np.random.choice(negative_indices, size=num_negative_to_keep, replace=False)
    
    # Combine positive and sampled negative windows
    keep_indices = np.concatenate([positive_indices, sampled_negative_indices])
    keep_indices = np.sort(keep_indices)
    
    return X[keep_indices], y[keep_indices]

def save_dataset(X, y, save_dir, name):
    """Save X and y tensors as PyTorch .pt file."""
    if len(X) == 0:
        print(f"Warning: No data to save for {name}")
        return
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Transpose X to have shape (batch_size, features, time_steps)
    X_tensor = X_tensor.transpose(1, 2)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{name}.pt")
    torch.save((X_tensor, y_tensor), save_path)
    print(f"Saved {name} dataset with shape X: {X_tensor.shape}, y: {y_tensor.shape}")

def save_config(config_dict, save_dir, filename='data_config.toml'):
    """Save experiment configuration to TOML file."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert numpy arrays to lists for TOML serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        elif isinstance(obj, set):
            return list(obj)
        return obj
    
    config_dict = convert_numpy(config_dict)
    
    with open(os.path.join(save_dir, filename), "w") as f:
        toml.dump(config_dict, f)

def resample(df):
    """Resample dataframe to consistent sampling rate (placeholder)."""
    print("RESAMPLE has not been implemented yet - you need to implement the function")
    raise RuntimeError("The resample function has not been implemented")
    return df

def process_session(session, project_path, config):
    """Process a single session and return windowed data."""
    df = combine_sensor_data(session, project_path, config.get('use_gyro', False))
    
    if df.empty:
        return np.array([]), np.array([])
    
    # Check for gaps and split if necessary
    segments = check_for_gaps(df, config.get('threshold_gap_minutes', 30))
    
    all_windows_X = []
    all_windows_y = []
    
    for segment in segments:
        # Apply resampling if configured
        if config.get('resample', False):
            segment = resample(segment)
        
        # Apply labels
        segment = apply_labels_to_df(segment, session, config['target_labels'], config.get('label_value', 1))
        
        # Create windows
        windows_X, windows_y = create_windows(
            segment, 
            config['window_size'], 
            config['step_size'], 
            config.get('use_gyro', False)
        )
        
        if len(windows_X) > 0:
            all_windows_X.append(windows_X)
            all_windows_y.append(windows_y)
    
    if not all_windows_X:
        return np.array([]), np.array([])
    
    # Concatenate all segments
    combined_X = np.concatenate(all_windows_X, axis=0)
    combined_y = np.concatenate(all_windows_y, axis=0)
    
    return combined_X, combined_y

def make_dataset_from_participants(participant_ids, config):
    """Create dataset from participant IDs using database."""
    all_X = []
    all_y = []
    
    for participant_id in participant_ids:
        print(f"Processing participant {participant_id}")
        
        projects = get_participant_projects(participant_id)
        for project_name in projects:
            raw_dataset_path = get_raw_dataset_path(project_name)
            sessions = get_sessions_for_project(project_name)
            
            # Filter sessions that have target labels
            sessions = [s for s in sessions if any(b.get('label') in config['target_labels'] for b in s.get('bouts', []))]
            
            for session in sessions:
                X, y = process_session(session, raw_dataset_path, config)
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
    
    if not all_X:
        return np.array([]), np.array([])
    
    # Concatenate all participants
    dataset_X = np.concatenate(all_X, axis=0)
    dataset_y = np.concatenate(all_y, axis=0)
    
    # Apply negative sampling
    dataset_X, dataset_y = filter_negative_windows(
        dataset_X, dataset_y, 
        config.get('percent_negative_windows', 1.0),
        config.get('random_seed')
    )
    
    # Shuffle dataset
    np.random.seed(config.get('random_seed'))
    indices = np.random.permutation(len(dataset_X))
    dataset_X = dataset_X[indices]
    dataset_y = dataset_y[indices]
    
    print(f"Dataset created with {len(dataset_X):,} windows")
    
    return dataset_X, dataset_y

def make_dataset_from_sessions(sessions_list, config):
    """Create dataset from list of (session, project_name, project_path) tuples."""
    all_X = []
    all_y = []
    
    for session, project_name, project_path in sessions_list:
        X, y = process_session(session, project_path, config)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        return np.array([]), np.array([])
    
    # Concatenate all sessions
    dataset_X = np.concatenate(all_X, axis=0)
    dataset_y = np.concatenate(all_y, axis=0)
    
    # Apply negative sampling and shuffling
    dataset_X, dataset_y = filter_negative_windows(
        dataset_X, dataset_y, 
        config.get('percent_negative_windows', 1.0),
        config.get('random_seed')
    )
    
    np.random.seed(config.get('random_seed'))
    indices = np.random.permutation(len(dataset_X))
    dataset_X = dataset_X[indices]
    dataset_y = dataset_y[indices]
    
    print(f"Dataset created with {len(dataset_X):,} windows")
    
    return dataset_X, dataset_y

def make_dataset_from_kfold_split(fold_sessions, config):
    """Create dataset from k-fold session split."""
    return make_dataset_from_sessions(fold_sessions, config)

def get_all_sessions_from_db(labeling_filter=None):
    """Get all sessions from database with optional label filtering."""
    all_sessions = []
    
    with db_connection() as conn:
        cursor = conn.cursor()
        query = "SELECT DISTINCT project_name FROM projects"
        cursor.execute(query)
        project_names = [row[0] for row in cursor.fetchall()]
    
    for project_name in project_names:
        try:
            sessions = get_sessions_for_project(project_name)
            raw_dataset_path = get_raw_dataset_path(project_name)
            
            for session in sessions:
                # Filter for sessions with target labels if specified
                if labeling_filter is None or any(
                    b.get('label') in labeling_filter 
                    for b in session.get('bouts', [])
                ):
                    all_sessions.append((session, project_name, raw_dataset_path))
        except Exception as e:
            print(f"Warning: Could not process project {project_name}: {e}")
            continue
    
    return all_sessions

def get_participant_sessions(participant_id, labeling_filter=None):
    """Get sessions for a specific participant with optional label filtering."""
    participant_sessions = []
    
    projects = get_participant_projects(participant_id)
    for project_name in projects:
        try:
            sessions = get_sessions_for_project(project_name)
            raw_dataset_path = get_raw_dataset_path(project_name)
            
            for session in sessions:
                # Filter for sessions with target labels if specified
                if labeling_filter is None or any(
                    b.get('label') in labeling_filter 
                    for b in session.get('bouts', [])
                ):
                    participant_sessions.append((session, project_name, raw_dataset_path))
        except Exception as e:
            print(f"Warning: Could not process project {project_name}: {e}")
            continue
    
    return participant_sessions

def create_base_config(**kwargs):
    """Create base configuration dictionary with default values."""
    default_config = {
        'target_labels': ['puff', 'puffs'],
        'window_size': 1024,
        'step_size': 1024,
        'use_gyro': False,
        'random_seed': 70,
        'percent_negative_windows': 1.0,
        'threshold_gap_minutes': 5,
        'label_value': 1,
        'resample': False
    }
    
    # Update with any provided kwargs
    default_config.update(kwargs)
    return default_config

def validate_config(config):
    """Validate configuration parameters."""
    required_keys = ['target_labels', 'window_size', 'step_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Configuration missing required key: {key}")
    
    if config['window_size'] <= 0:
        raise ValueError("window_size must be positive")
    
    if config['step_size'] <= 0:
        raise ValueError("step_size must be positive")
    
    if not (0 <= config.get('percent_negative_windows', 1.0) <= 1.0):
        raise ValueError("percent_negative_windows must be between 0 and 1")