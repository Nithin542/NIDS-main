import os
import pandas as pd
import joblib
import sqlite3
import numpy as np
import json
from nfstream import NFStreamer
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

print("Loading model & assets...")

# ==============================
# Load Model Assets
# ==============================
try:
    model = joblib.load('nids_model.pkl')
    scaler = joblib.load('scaler.pkl')
    le = joblib.load('label_encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    print("Assets loaded successfully.")
except Exception as e:
    print(f"❌ Error loading assets: {e}")
    exit()

# ==============================
# Database Setup (macOS Safe)
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "nids.db")

def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                src_ip TEXT,
                dst_ip TEXT,
                attack_type TEXT,
                confidence REAL,
                features_json TEXT
            )
        """)
        try:
            cursor.execute("ALTER TABLE logs ADD COLUMN features_json TEXT;")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        print("✅ Database initialized.")
        return conn
    except Exception as e:
        print(f"❌ Database Error: {e}")
        return None

db_conn = init_db()

def save_log_entry(conn, data):
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (timestamp, src_ip, dst_ip, attack_type, confidence, features_json) VALUES (?, ?, ?, ?, ?, ?)",
            (data['timestamp'], data['src_ip'], data['dst_ip'], data['attack_type'], data['confidence'], data['features_json'])
        )
        conn.commit()
    except Exception as e:
        print(f"DB Insert Error: {e}")

# ==============================
# CICIDS → NFStream Mapping
# ==============================
MAPS = {
    'Destination Port': 'dst_port',
    'Flow Duration': 'bidirectional_duration_ms',

    'Total Fwd Packets': 'src2dst_packets',
    'Total Length of Fwd Packets': 'src2dst_bytes',
    'Fwd Packet Length Max': 'src2dst_max_ps',
    'Fwd Packet Length Min': 'src2dst_min_ps',
    'Fwd Packet Length Mean': 'src2dst_mean_ps',
    'Fwd Packet Length Std': 'src2dst_stddev_ps',

    'Bwd Packet Length Max': 'dst2src_max_ps',
    'Bwd Packet Length Min': 'dst2src_min_ps',
    'Bwd Packet Length Mean': 'dst2src_mean_ps',
    'Bwd Packet Length Std': 'dst2src_stddev_ps',

    'Flow Bytes/s': 'bidirectional_bytes',
    'Flow Packets/s': 'bidirectional_packets',

    'Flow IAT Mean': 'bidirectional_mean_piat_ms',
    'Flow IAT Std': 'bidirectional_stddev_piat_ms',
    'Flow IAT Max': 'bidirectional_max_piat_ms',
    'Flow IAT Min': 'bidirectional_min_piat_ms',

    'Fwd IAT Total': 'src2dst_duration_ms',
    'Fwd IAT Mean': 'src2dst_mean_piat_ms',
    'Fwd IAT Std': 'src2dst_stddev_piat_ms',
    'Fwd IAT Max': 'src2dst_max_piat_ms',
    'Fwd IAT Min': 'src2dst_min_piat_ms',

    'Bwd IAT Total': 'dst2src_duration_ms',
    'Bwd IAT Mean': 'dst2src_mean_piat_ms',
    'Bwd IAT Std': 'dst2src_stddev_piat_ms',
    'Bwd IAT Max': 'dst2src_max_piat_ms',
    'Bwd IAT Min': 'dst2src_min_piat_ms',

    'Fwd Header Length': 'src2dst_header_size',
    'Bwd Header Length': 'dst2src_header_size',

    'Min Packet Length': 'bidirectional_min_ps',
    'Max Packet Length': 'bidirectional_max_ps',
    'Packet Length Mean': 'bidirectional_mean_ps',
    'Packet Length Std': 'bidirectional_stddev_ps',

    'FIN Flag Count': 'bidirectional_fin_packets',
    'PSH Flag Count': 'bidirectional_psh_packets',
    'ACK Flag Count': 'bidirectional_ack_packets',

    'Average Packet Size': 'bidirectional_mean_ps',
    'Subflow Fwd Bytes': 'src2dst_bytes',
    'Init_Win_bytes_forward': 'src2dst_init_window_size',
    'Init_Win_bytes_backward': 'dst2src_init_window_size',
}

# ==============================
# NFStreamer Setup (macOS)
# ==============================
print("Starting NFStreamer... (Requires sudo)")

streamer = NFStreamer(
    source="en0",   # macOS WiFi interface
    statistical_analysis=True,
    idle_timeout=1,
    active_timeout=1,
    promiscuous_mode=True
)

print("🛡️ NetShield NIDS Engine is LIVE...")

# ==============================
# Real-Time Detection Loop
# ==============================
for flow in streamer:
    try:
        features_dict = {}

        for col in feature_names:
            if col == 'Attack Type':
                continue

            attr_name = MAPS.get(col)
            val = getattr(flow, attr_name, 0) if attr_name else 0

            # Convert durations to microseconds if required
            if "Duration" in col or "IAT" in col:
                val = val * 1000

            if col == 'Packet Length Variance':
                std_val = getattr(flow, 'bidirectional_stddev_ps', 0)
                val = std_val ** 2

            features_dict[col] = [val]

        # Convert to DataFrame
        X_df = pd.DataFrame(features_dict)

        # Scale
        X_scaled = scaler.transform(X_df)

        # Predict
        probs = model.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        label = le.inverse_transform([pred_idx])[0]

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save to DB
        log_entry = {
            'timestamp': now,
            'src_ip': flow.src_ip,
            'dst_ip': flow.dst_ip,
            'attack_type': label,
            'confidence': float(confidence),
            'features_json': json.dumps(X_scaled.tolist()[0])
        }

        save_log_entry(db_conn, log_entry)

        # Console Output
        icon = "🟢" if label == "Normal Traffic" else "⚠️"
        print(f"{icon} [{now}] {label} | {flow.src_ip} → {flow.dst_ip} | Conf: {confidence*100:.2f}%")

    except Exception as e:
        print(f"Flow Processing Error: {e}")
        continue