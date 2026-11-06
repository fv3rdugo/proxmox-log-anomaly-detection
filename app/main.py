from fastapi import FastAPI, Request
import uvicorn

from sklearn.ensemble import IsolationForest

import numpy as np
import requests
import joblib
import json
import re
import time

from datetime import datetime, timezone
from pathlib import Path
from collections import deque, defaultdict
from threading import Lock

# Drain3
from drain3 import TemplateMiner
from drain3.file_persistence import FilePersistence
from drain3.template_miner_config import TemplateMinerConfig

# ============================================================
# CONFIGURACIÓ
# ============================================================

app = FastAPI(
    title="Proxmox Log Anomaly Detection",
    version="2.2"
)

# ============================================================
# OPENSEARCH
# ============================================================

OPENSEARCH_URL = "http://opensearch:9200"
OPENSEARCH_INDEX = "logs-anomalies"
OPENSEARCH_BULK_URL = (
    f"{OPENSEARCH_URL}/{OPENSEARCH_INDEX}/_bulk"
)

# ============================================================
# PERSISTÈNCIA
# ============================================================

MODEL_DIR = Path("/app/model")
MODEL_PATH = (
    MODEL_DIR /
    "isolation_forest.joblib"
)

WARMUP_PATH = (
    MODEL_DIR /
    "warmup_data.json"
)

# ============================================================
# DRAIN3
# ============================================================

DRAIN_DIR = Path("/app/drain")
DRAIN_STATE_PATH = (
    DRAIN_DIR /
    "drain_state.bin"
)

DRAIN_CONFIG_PATH = Path(
    "/app/drain3.ini"
)

# ============================================================
# MACHINE LEARNING
# ============================================================

WARMUP_SIZE = 300
CONTAMINATION = 0.05
N_ESTIMATORS = 200

# ============================================================
# FEATURES TEMPORALS
# ============================================================

WINDOW_SECONDS = 60
MAX_RECENT_EVENTS = 10000

# ============================================================
# OPENSEARCH
# ============================================================

OPENSEARCH_TIMEOUT = 10
OPENSEARCH_RETRIES = 3

# ============================================================
# ESTAT GLOBAL
# ============================================================

model = None
warmup_data = []
model_lock = Lock()

recent_events = deque(
    maxlen=MAX_RECENT_EVENTS
)

template_total_count = defaultdict(
    int
)

template_miner = None
drain_lock = Lock()
recent_events_lock = Lock()

# ============================================================
# FEATURES DEL MODEL
# ============================================================

FEATURE_NAMES = [
    "log_length",
    "template_size",
    "parameter_count",
    "severity",
    "logs_per_minute",
    "errors_per_minute",
    "warnings_per_minute",
    "unique_templates_per_minute",
    "current_template_per_minute"
]

# ============================================================
# UTILITATS
# ============================================================

def utc_now():

    return datetime.now(
        timezone.utc
    )

def safe_float(value, default=0.0):
    try:
        return float(value)

    except Exception:
        return default

# ============================================================
# TIMESTAMP
# ============================================================

def parse_timestamp(value):

    """
    Intenta convertir diferents formats de timestamp
    a datetime UTC.

    Si no es pot interpretar, retorna None.
    """

    if value is None:
        return None

    if isinstance(
        value,
        datetime
    ):

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value.astimezone(
            timezone.utc
        )

    # Unix timestamp

    if isinstance(
        value,
        (int, float)
    ):

        try:
            return datetime.fromtimestamp(
                float(value),
                tz=timezone.utc
            )

        except Exception:
            return None

    value = str(
        value
    ).strip()

    if not value:
        return None

    # ISO 8601

    try:
        normalized = value.replace(
            "Z",
            "+00:00"
        )

        parsed = datetime.fromisoformat(
            normalized
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed.astimezone(
            timezone.utc
        )

    except Exception:
        pass

    # Formats habituals

    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%b %d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S",
    ]

    for fmt in formats:
        try:
            parsed = datetime.strptime(
                value,
                fmt
            )

            return parsed.replace(
                tzinfo=timezone.utc
            )

        except Exception:
            continue

    return None

def get_event_time(log):

    """
    Retorna el timestamp del log.

    Si no existe o és invàlid,
    utilitza el timestamp actual.
    """

    timestamp_value = (
        log.get("timestamp")
        or log.get("@timestamp")
        or log.get("date")
    )

    parsed = parse_timestamp(
        timestamp_value
    )

    if parsed is not None:
        return parsed

    return utc_now()

# ============================================================
# DRAIN3
# ============================================================

def initialize_drain3():
    global template_miner

    DRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    config = TemplateMinerConfig()

    if DRAIN_CONFIG_PATH.exists():
        try:
            config.load(
                str(DRAIN_CONFIG_PATH)
            )

            print(
                "[DRAIN3] Configuració carregada."
            )
        except Exception as e:
            print(
                f"[DRAIN3] Error carregant config: {e}"
            )
    else:
        print(
            "[DRAIN3] No s'ha trobat drain3.ini. "
            "Utilitzant configuració per defecte."
        )

    persistence = FilePersistence(
        str(DRAIN_STATE_PATH)
    )

    template_miner = TemplateMiner(
        persistence_handler=persistence,
        config=config
    )

    print(
        "[DRAIN3] TemplateMiner inicialitzat."
    )

def extract_parameters(
    message,
    template
):

    if not template:
        return []

    escaped = re.escape(
        template
    )

    pattern = escaped.replace(
        r"\<\*\>",
        r"(.*?)"
    )

    try:
        match = re.match(
            "^" +
            pattern +
            "$",
            message
        )

        if not match:
            return []

        return list(
            match.groups()
        )

    except Exception:
        return []

def process_with_drain3(
    message
):

    if template_miner is None:
        raise RuntimeError(
            "Drain3 no està inicialitzat."
        )
    with drain_lock:
        result = (
            template_miner
            .add_log_message(
                message
            )
        )
        cluster_id = (
            result.get(
                "cluster_id"
            )
        )
        template = (
            result.get(
                "template_mined"
            )
        )
        change_type = (
            result.get(
                "change_type"
            )
        )
        cluster_size = 0
        try:
            cluster = (
                template_miner
                .drain
                .clusters[
                    cluster_id
                ]
            )
            cluster_size = (
                cluster.size
            )
        except Exception:
            pass
        parameters = (
            extract_parameters(
                message,
                template
            )
        )
        return {
            "cluster_id":
                cluster_id,
            "template":
                template,
            "parameters":
                parameters,
            "parameter_count":
                len(parameters),
            "cluster_size":
                cluster_size,
            "change_type":
                change_type
        }

# ============================================================
# SEVERITY
# ============================================================

SEVERITY_MAP = {
    "emerg": 7,
    "alert": 6,
    "crit": 5,
    "critical": 5,
    "err": 4,
    "error": 4,
    "warning": 3,
    "warn": 3,
    "notice": 2,
    "info": 1,
    "debug": 0
}

def extract_severity(log):
    value = (
        log.get("severity")
        or log.get("level")
        or log.get("PRIORITY")
        or ""
    )

    value = str(
        value
    ).lower().strip()

    if value.isdigit():
        try:
            priority = int(
                value
            )
            return max(
                0,
                7 - priority
            )
        except Exception:
            pass

    return SEVERITY_MAP.get(
        value,
        1
    )

def severity_name(value):
    names = {
        7: "emerg",
        6: "alert",
        5: "critical",
        4: "error",
        3: "warning",
        2: "notice",
        1: "info",
        0: "debug"
    }

    return names.get(
        value,
        "unknown"
    )

# ============================================================
# FEATURES TEMPORALS
# ============================================================

def cleanup_recent_events(
    now
):
    limit = (
        now.timestamp()
        -
        WINDOW_SECONDS
    )

    while recent_events:
        first = (
            recent_events[0]
        )
        if (
            first["timestamp"]
            .timestamp()
            >= limit
        ):
            break
        recent_events.popleft()

def calculate_temporal_features(
    now,
    current_cluster_id,
    severity
):

    with recent_events_lock:
        cleanup_recent_events(
            now
        )
        events = list(
            recent_events
        )

    total_logs = len(
        events
    )
    errors = sum(
        1
        for event in events
        if event["severity"] >= 4

    )

    warnings = sum(
        1
        for event in events
        if event["severity"] == 3
    )

    unique_templates = len(
        set(
            event["template_id"]
            for event in events
        )
    )

    current_total = (
        total_logs + 1
    )

    current_errors = (
        errors
        +
        (
            1
            if severity >= 4
            else 0
        )
    )

    current_warnings = (
        warnings
        +
        (
            1
            if severity == 3
            else 0
        )
    )

    current_unique_templates = (
        unique_templates
    )

    if current_cluster_id is not None:
        current_unique_templates = len(
            set(
                [
                    event["template_id"]
                    for event in events
                ]
                +
                [current_cluster_id]
            )
        )

    current_template_count = sum(
        1
        for event in events
        if (
            event["template_id"]
            ==
            current_cluster_id
        )
    )

    current_template_count += 1

    return {
        "logs_per_minute":
            current_total,
        "errors_per_minute":
            current_errors,
        "warnings_per_minute":
            current_warnings,
        "unique_templates_per_minute":
            current_unique_templates,
        "current_template_per_minute":
            current_template_count
    }

def register_recent_event(
    timestamp,
    cluster_id,
    severity
):

    with recent_events_lock:
        recent_events.append(
            {
                "timestamp":
                    timestamp,
                "template_id":
                    cluster_id,
                "severity":
                    severity
            }
        )

# ============================================================
# MACHINE LEARNING
# ============================================================

def create_model():
    return IsolationForest(
        n_estimators=
            N_ESTIMATORS,
        contamination=
            CONTAMINATION,
        random_state=
            42,
        n_jobs=
            -1
    )

def save_warmup_data():

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        with open(
            WARMUP_PATH,
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(
                warmup_data,
                f
            )
        return True

    except Exception as e:
        print(
            f"[ML] Error guardant warm-up: {e}"
        )
        return False

def load_warmup_data():
    global warmup_data
    if not WARMUP_PATH.exists():
        return False
    try:
        with open(
            WARMUP_PATH,
            "r",
            encoding="utf-8"
        ) as f:
            data = json.load(
                f
            )
        if not isinstance(
            data,
            list
        ):
            return False
        warmup_data = data
        print(
            f"[ML] Warm-up recuperat: "
            f"{len(warmup_data)} mostres."
        )
        return True

    except Exception as e:
        print(
            f"[ML] Error carregant warm-up: {e}"
        )
        warmup_data = []
        return False

def train_model(
    data
):

    global model
    if len(data) < WARMUP_SIZE:
        print(
            f"[ML] Dades insuficients: "
            f"{len(data)}/"
            f"{WARMUP_SIZE}"
        )
        return False

    X = np.array(
        data,
        dtype=float
    )

    if X.ndim != 2:
        print(
            "[ML] Dataset invàlid."
        )
        return False

    if X.shape[1] != len(
        FEATURE_NAMES
    ):
        print(
            "[ML] Nombre de features "
            "incorrecte: "
            f"{X.shape[1]} "
            f"vs "
            f"{len(FEATURE_NAMES)}"
        )
        return False

    print(
        "[ML] ================================"
    )

    print(
        f"[ML] Entrenant Isolation Forest "
        f"amb {len(X)} mostres."
    )

    print(
        f"[ML] Features: {X.shape[1]}"
    )

    new_model = create_model()

    try:
        new_model.fit(
            X
        )

    except Exception as e:
        print(
            f"[ML] Error durant training: {e}"
        )
        return False

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    try:
        joblib.dump(
            new_model,
            MODEL_PATH
        )

    except Exception as e:
        print(
            f"[ML] Error guardant model: {e}"
        )
        return False

    model = new_model

    print(
        f"[ML] Model guardat a {MODEL_PATH}"
    )

    print(
        "[ML] ================================"
    )
    return True


def load_model():
    global model
    if not MODEL_PATH.exists():
        print(
            "[ML] No existeix model persistent."
        )

        return False

    try:
        loaded_model = joblib.load(
            MODEL_PATH
        )

        # Validació bàsica
        if not hasattr(
            loaded_model,
            "predict"
        ):
            raise ValueError(
                "Model invàlid."
            )
        model = loaded_model
        print(
            f"[ML] Model carregat: "
            f"{MODEL_PATH}"
        )
        return True

    except Exception as e:
        print(
            f"[ML] Error carregant model: {e}"
        )
        model = None
        return False

def remove_warmup_file():
    try:
        if WARMUP_PATH.exists():
            WARMUP_PATH.unlink()

    except Exception as e:
        print(
            f"[ML] Error eliminant warm-up: {e}"
        )

# ============================================================
# OPENSEARCH BULK
# ============================================================

def send_bulk_to_opensearch(
    logs,
    retries=OPENSEARCH_RETRIES
):

    """
    Envia tots els logs del batch a OpenSearch
    mitjançant una única petició _bulk.

    Retorna True només si OpenSearch accepta
    tots els documents.
    """

    if not logs:
        return True
    bulk_lines = []

    for log in logs:
        bulk_lines.append(
            json.dumps(
                {
                    "index": {}
                }
            )
        )

        bulk_lines.append(
            json.dumps(
                log,
                ensure_ascii=False
            )
        )

    bulk_body = (
        "\n".join(
            bulk_lines
        )
        +
        "\n"
    )

    headers = {
        "Content-Type":
            "application/x-ndjson"
    }

    for attempt in range(
        1,
        retries + 1
    ):

        try:
            response = requests.post(
                OPENSEARCH_BULK_URL,
                data=
                    bulk_body.encode(
                        "utf-8"
                    ),
                headers=
                    headers,
                timeout=
                    OPENSEARCH_TIMEOUT
            )

            if not response.ok:
                print(
                    f"[OpenSearch] Bulk HTTP "
                    f"{response.status_code}: "
                    f"{response.text[:500]}"
                )

            else:
                result = response.json()
                if result.get(
                    "errors"
                ):
                    failed = 0
                    for item in result.get(
                        "items",
                        []
                    ):
                        index_result = (
                            item.get(
                                "index",
                                {}
                            )
                        )
                        if "error" in (
                            index_result
                        ):
                            failed += 1
                    print(
                        f"[OpenSearch] Bulk "
                        f"completat amb "
                        f"{failed} errors."
                    )

                    if failed == 0:
                        return True
                else:
                    print(
                        f"[OpenSearch] Bulk OK: "
                        f"{len(logs)} documents."
                    )
                    return True
        except (
            requests.exceptions.ConnectionError
        ):
            print(
                f"[OpenSearch] Connexió "
                f"fallida "
                f"(intent "
                f"{attempt}/"
                f"{retries})"
            )

        except (
            requests.exceptions.Timeout
        ):
            print(
                f"[OpenSearch] Timeout "
                f"(intent "
                f"{attempt}/"
                f"{retries})"
            )

        except Exception as e:
            print(
                f"[OpenSearch] Error: {e}"
            )
        if attempt < retries:
            time.sleep(
                2
            )
    print(
        "[OpenSearch] No s'ha pogut "
        f"enviar el batch de "
        f"{len(logs)} documents."
    )
    return False

# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status":
            "ok",
        "model_ready":
            model is not None,
        "drain3_ready":
            template_miner is not None
    }

# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup_event():
    global warmup_data
    print("=" * 60)
    print(
        "PROXMOX LOG ANOMALY DETECTION v2.2"
    )
    print("=" * 60)
    initialize_drain3()
    load_warmup_data()
    if load_model():
        print(
            "[ML] Model disponible."
        )
        print(
            "[ML] Mode DETECCIÓ activat."
        )
    else:
        print(
            "[ML] No hi ha model."
        )
        print(
            "[ML] Activant WARM-UP."
        )
        if warmup_data:
            print(
                f"[ML] Warm-up actual: "
                f"{len(warmup_data)}/"
                f"{WARMUP_SIZE}"
            )
    print("=" * 60)

# ============================================================
# DETECT
# ============================================================

@app.post("/detect")
async def detect(
    request: Request
):

    global model
    global warmup_data

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    try:
        logs = await request.json()

    except Exception:
        return {
            "status":
                "error",
            "message":
                "Invalid JSON"
        }

    if isinstance(
        logs,
        dict
    ):
        logs = [logs]

    if not isinstance(
        logs,
        list
    ):
        return {
            "status":
                "error",
            "message":
                "Expected JSON object or list"
        }

    processed_logs = []
    processed = 0
    anomalies = 0
    opensearch_ok = True

    # --------------------------------------------------------
    # Processament
    # --------------------------------------------------------

    for log in logs:

        if not isinstance(
            log,
            dict
        ):

            continue

        # ====================================================
        # MISSATGE
        # ====================================================

        text_log = (
            log.get("message")
            or log.get("msg")
            or log.get("MESSAGE")
            or ""
        )

        text_log = str(
            text_log
        )

        # ====================================================
        # TIMESTAMP
        # ====================================================

        event_time = get_event_time(
            log
        )

        log["timestamp"] = (
            event_time.isoformat()
        )

        # ====================================================
        # LONGITUD
        # ====================================================

        log_length = len(
            text_log
        )

        log["log_length"] = (
            log_length
        )

        # Compatibilitat Grafana

        log["valor_metric"] = (
            float(log_length)
        )

        # ====================================================
        # SEVERITY
        # ====================================================

        severity = extract_severity(
            log
        )

        log["severity_numeric"] = (
            severity
        )

        log["severity_name"] = (
            severity_name(
                severity
            )
        )

        # ====================================================
        # DRAIN3
        # ====================================================

        try:
            drain_result = (
                process_with_drain3(
                    text_log
                )
            )
            cluster_id = (
                drain_result[
                    "cluster_id"
                ]
            )
            template = (
                drain_result[
                    "template"
                ]
            )
            parameters = (
                drain_result[
                    "parameters"
                ]
            )
            parameter_count = (
                drain_result[
                    "parameter_count"
                ]
            )
            cluster_size = (
                drain_result[
                    "cluster_size"
                ]
            )
            change_type = (
                drain_result[
                    "change_type"
                ]
            )

        except Exception as e:
            print(
                f"[DRAIN3] Error: {e}"
            )
            cluster_id = -1
            template = text_log
            parameters = []
            parameter_count = 0
            cluster_size = 0
            change_type = None

        # ----------------------------------------------------
        # Drain3 fields
        # ----------------------------------------------------

        log["drain_cluster_id"] = (
            cluster_id
        )

        log["drain_template"] = (
            template
        )

        log["drain_parameters"] = (
            parameters
        )

        log["drain_parameter_count"] = (
            parameter_count
        )

        log["drain_cluster_size"] = (
            cluster_size
        )

        log["drain_change_type"] = (
            change_type
        )

        template_total_count[
            cluster_id
        ] += 1

        log[
            "drain_template_total_count"
        ] = (
            template_total_count[
                cluster_id
            ]
        )

        # ====================================================
        # FEATURES TEMPORALS
        # ====================================================

        temporal = (
            calculate_temporal_features(
                event_time,
                cluster_id,
                severity
            )
        )

        log.update(
            temporal
        )

        # ====================================================
        # FEATURE VECTOR
        # ====================================================

        feature_vector = [
            float(
                log_length
            ),
            float(
                cluster_size
            ),
            float(
                parameter_count
            ),
            float(
                severity
            ),
            float(
                temporal[
                    "logs_per_minute"
                ]
            ),
            float(
                temporal[
                    "errors_per_minute"
                ]
            ),
            float(
                temporal[
                    "warnings_per_minute"
                ]
            ),
            float(
                temporal[
                    "unique_templates_per_minute"
                ]
            ),
            float(
                temporal[
                    "current_template_per_minute"
                ]
            )
        ]

        # ====================================================
        # FEATURES LEGIBLES
        # ====================================================

        log["ml_features"] = {
            "log_length":
                float(log_length),
            "template_size":
                float(cluster_size),
            "parameter_count":
                float(parameter_count),
            "severity":
                float(severity),
            "logs_per_minute":
                float(
                    temporal[
                        "logs_per_minute"
                    ]
                ),
            "errors_per_minute":
                float(
                    temporal[
                        "errors_per_minute"
                    ]
                ),
            "warnings_per_minute":
                float(
                    temporal[
                        "warnings_per_minute"
                    ]
                ),
            "unique_templates_per_minute":
                float(
                    temporal[
                        "unique_templates_per_minute"
                    ]
                ),
            "current_template_per_minute":
                float(
                    temporal[
                        "current_template_per_minute"
                    ]
                )

        }

        log["ml_feature_names"] = (
            FEATURE_NAMES
        )

        # ====================================================
        # MACHINE LEARNING
        # ====================================================

        with model_lock:

            # ------------------------------------------------
            # WARM-UP
            # ------------------------------------------------

            if model is None:

                warmup_data.append(
                    feature_vector
                )

                save_warmup_data()

                log["is_anomaly"] = (
                    False
                )

                log["anomaly_score"] = (
                    None
                )

                log["ml_prediction"] = (
                    0
                )

                current = len(
                    warmup_data
                )

                if (
                    current % 25 == 0
                    or current == 1
                ):
                    print(
                        f"[ML] Warm-up "
                        f"{current}/"
                        f"{WARMUP_SIZE}"
                    )

                if current >= WARMUP_SIZE:
                    print(
                        "[ML] Warm-up complet."
                    )
                    success = train_model(
                        warmup_data
                    )
                    if success:
                        warmup_data = []
                        remove_warmup_file()

            # ------------------------------------------------
            # DETECCIÓ
            # ------------------------------------------------

            else:
                X = np.array(
                    [feature_vector],
                    dtype=float
                )

                try:
                    prediction = (
                        model
                        .predict(X)[0]
                    )

                    score = (
                        model
                        .decision_function(X)[0]
                    )

                    is_anomaly = (
                        prediction == -1
                    )

                    log["is_anomaly"] = (
                        bool(is_anomaly)
                    )

                    log["anomaly_score"] = (
                        float(score)
                    )

                    log["ml_prediction"] = (
                        int(prediction)
                    )

                    # Score més baix =
                    # més anòmal.

                    if score < -0.20:
                        anomaly_level = (
                            "critical"
                        )

                    elif score < -0.10:
                        anomaly_level = (
                            "high"
                        )

                    elif score < 0:
                        anomaly_level = (
                            "medium"
                        )

                    else:
                        anomaly_level = (
                            "normal"
                        )

                    log["anomaly_level"] = (
                        anomaly_level
                    )

                    if is_anomaly:
                        anomalies += 1
                        print(
                            "[ML] 🚨 ANOMALIA"
                        )
                        print(
                            f"     template="
                            f"{cluster_id}"
                        )

                        print(
                            f"     score="
                            f"{score:.5f}"
                        )

                        print(
                            f"     template/min="
                            f"{temporal['current_template_per_minute']}"
                        )

                        print(
                            f"     severity="
                            f"{severity_name(severity)}"

                        )

                        print(
                            f"     message="
                            f"{text_log[:200]}"
                        )

                except Exception as e:
                    print(
                        f"[ML] Error prediction: {e}"
                    )

                    log["is_anomaly"] = (
                        False
                    )

                    log["anomaly_score"] = (
                        None
                    )

                    log["ml_prediction"] = (
                        0
                    )

                    log["anomaly_level"] = (
                        "unknown"
                    )

        # ====================================================
        # REGISTRAR EVENT
        # ====================================================

        register_recent_event(
            event_time,
            cluster_id,
            severity
        )

        # ====================================================
        # DOCUMENT
        # ====================================================

        processed_logs.append(
            log
        )

        processed += 1

    # ========================================================
    # OPENSEARCH BULK
    # ========================================================

    if processed_logs:
        opensearch_ok = (
            send_bulk_to_opensearch(
                processed_logs
            )
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {
        "status":
            "processed",
        "processed":
            processed,
        "anomalies":
            anomalies,
        "opensearch":
            opensearch_ok,
        "model_ready":
            model is not None,
        "warmup_progress": (
            len(warmup_data)
            if model is None
            else None
        )
    }

# ============================================================
# STATUS
# ============================================================

@app.get("/status")
def status():
    template_count = 0
    try:
        with drain_lock:
            template_count = len(
                template_miner
                .drain
                .clusters
            )

    except Exception:
        pass

    with recent_events_lock:
        recent_count = len(
            recent_events
        )

    return {
        "application":
            "Proxmox Log Anomaly Detection",
        "version":
            "2.2",
        "model_ready":
            model is not None,
        "model_path":
            str(MODEL_PATH),
        "warmup_size":
            WARMUP_SIZE,
        "warmup_current": (
            len(warmup_data)
            if model is None
            else WARMUP_SIZE
        ),
        "contamination":
            CONTAMINATION,
        "n_estimators":
            N_ESTIMATORS,
        "feature_count":
            len(FEATURE_NAMES),
        "features":
            FEATURE_NAMES,
        "drain3_ready":
            template_miner is not None,
        "drain3_clusters":
            template_count,
        "recent_events":
            recent_count
    }

# ============================================================
# DRAIN3 TEMPLATES
# ============================================================

@app.get("/templates")
def templates():

    if template_miner is None:
        return {
            "status":
                "error",
            "message":
                "Drain3 not initialized"
        }

    clusters = []

    try:
        with drain_lock:
            for cluster in (
                template_miner
                .drain
                .clusters
            ):
                clusters.append(
                    {
                        "cluster_id":
                            cluster.cluster_id,
                        "size":
                            cluster.size,
                        "template":
                            cluster.get_template()
                    }
                )

    except Exception as e:

        return {
            "status":
                "error",
            "message":
                str(e)
        }

    clusters.sort(
        key=lambda x:
            x["size"],
        reverse=True
    )

    return {
        "status":
            "ok",
        "count":
            len(clusters),
        "clusters":
            clusters
    }

# ============================================================
# TRAIN MANUAL
# ============================================================

@app.post("/train")
def manual_train():

    global warmup_data

    with model_lock:
        if len(
            warmup_data
        ) < WARMUP_SIZE:
            return {
                "status":
                    "error",
                "message":
                    "No hi ha prou dades "
                    "de warm-up.",
                "samples":
                    len(warmup_data),
                "required":
                    WARMUP_SIZE
            }

        success = train_model(
            warmup_data
        )

        if not success:
            return {
                "status":
                    "error",
                "message":
                    "Training failed"
            }

        samples = len(
            warmup_data
        )

        warmup_data = []

        remove_warmup_file()

    return {
        "status":
            "trained",
        "samples":
            samples,
        "model_path":
            str(MODEL_PATH)
    }

# ============================================================
# RESET MODEL
# ============================================================

@app.delete("/model")
def reset_model():

    global model
    global warmup_data

    with model_lock:
        model = None
        warmup_data = []
        if MODEL_PATH.exists():
            try:
                MODEL_PATH.unlink()
            except Exception as e:
                print(
                    f"[ML] Error eliminant model: {e}"
                )
        remove_warmup_file()

    print(
        "[ML] Model reset. "
        "Nou warm-up iniciat."
    )

    return {
        "status":
            "reset",
        "message":
            "Isolation Forest eliminat. "
            "El proper log iniciarà "
            "un nou warm-up."
    }

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )
