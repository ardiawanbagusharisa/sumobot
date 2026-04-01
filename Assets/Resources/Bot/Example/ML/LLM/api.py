import re
import platform
import time
from typing import Dict, Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
from pymilvus import connections, Collection

COLLECTION_NAME = "sumobot_states"
TOP_K = 1

def detect_gpu():
    """Detect if NVIDIA GPU with CUDA is available"""
    try:
        import pynvml
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            name = pynvml.nvmlDeviceGetName(handle)
            print(f"✅ Detected GPU: {name}")
            pynvml.nvmlShutdown()
            return True
    except:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ Detected GPU via PyTorch: {torch.cuda.get_device_name(0)}")
            return True
    except:
        pass
    print("No NVIDIA GPU detected, using CPU mode")
    return False

def get_search_params(has_gpu=None):
    """Get search parameters based on GPU availability"""
    if has_gpu is None:
        has_gpu = detect_gpu()
    return {"nprobe": 64 if has_gpu else 16}

HAS_GPU = detect_gpu()
SEARCH_PARAMS = get_search_params(HAS_GPU)
NPROBE = SEARCH_PARAMS["nprobe"]

print("="*70)
print("🤖 Sumobot Milvus Vector Search API Server")
print("="*70)
print(f"Platform: {platform.system()} {platform.machine()}")
print(f"Mode: {'GPU-Accelerated 🚀' if HAS_GPU else 'CPU'}")
print(f"Milvus: Lite (./milvus_sumobot.db)")
print(f"Collection: {COLLECTION_NAME}")
print(f"Search nprobe: {NPROBE}")
print("="*70)

print("\n⏳ Connecting to Milvus Lite...")
start_time = time.time()

try:
    connections.connect(uri="./milvus_sumobot.db")
    col = Collection(COLLECTION_NAME)
    col.load()

    load_time = time.time() - start_time
    num_entities = col.num_entities

    print(f"✅ Connected to Milvus in {load_time:.2f}s")
    print(f"📊 Collection has {num_entities} entities\n")

except Exception as e:
    print(f"❌ Failed to connect to Milvus: {e}")
    print("\nMake sure:")
    print(f"  1. milvus_sumobot.db exists (run train_llm.ipynb first)")
    print(f"  2. Collection '{COLLECTION_NAME}' exists")
    print("  3. pymilvus is installed: pip install pymilvus")
    exit(1)

def encode_state(angle: float, angle_score: float, dist_score: float,
                 near_score: float, facing: float) -> np.ndarray:
    angle_normalized = angle / 180.0
    return np.array([angle_normalized, angle_score, dist_score, near_score, facing], dtype=np.float32)

def parse_state_string(state_str: str) -> Dict[str, float]:
    state_dict = {}
    state_str = state_str.rstrip('.,;')

    for part in state_str.split(","):
        part = part.strip()
        if "=" in part:
            key, value = part.split("=", 1)
            value_clean = value.strip().rstrip('.')
            try:
                state_dict[key.strip()] = float(value_clean)
            except ValueError as e:
                raise ValueError(f"Cannot parse '{key.strip()}={value}' - invalid float value: {e}")

    return state_dict

def parse_action(output: str) -> Dict[str, Optional[float]]:
    action_map = {
        "SK": "Skill",
        "DS": "Dash",
        "FWD": "Accelerate",
        "TL": "TurnLeft",
        "TR": "TurnRight",
    }

    actions: Dict[str, Optional[float]] = {}

    for part in [p.strip() for p in output.split(",")]:
        if not part:
            continue

        name = part
        duration = None

        direct_match = re.match(r"^([A-Za-z]+)\s*([\d.]+)$", part)
        if direct_match:
            name = direct_match.group(1).strip()
            duration = float(direct_match.group(2))

        for short, full in action_map.items():
            if name.upper().startswith(short):
                name = full
                break

        actions[name] = duration

    return actions

def query_action(angle: float, angle_score: float, dist_score: float,
                 near_score: float, facing: float, top_k: int = TOP_K) -> dict:
    vec = encode_state(angle, angle_score, dist_score, near_score, facing)

    start = time.time()
    result = col.search(
        data=[vec.tolist()],
        anns_field="state_vec",
        param={"nprobe": NPROBE},
        limit=top_k,
        output_fields=["action"],
    )
    search_time = (time.time() - start) * 1000

    if len(result[0]) == 0:
        raise ValueError("No similar states found in database")

    top_hit = result[0][0]
    action_str = top_hit.entity.get("action")
    distance = top_hit.distance

    parsed_actions = parse_action(action_str)

    response = {
        "raw_output": action_str,
        "action": parsed_actions,
        "search_time_ms": round(search_time, 2),
        "distance": round(distance, 4)
    }

    if top_k > 1:
        response["top_k_results"] = [
            {
                "action": hit.entity.get("action"),
                "distance": round(hit.distance, 4)
            }
            for hit in result[0]
        ]

    return response

def query_action_from_string(state_str: str, top_k: int = TOP_K) -> dict:
    state_dict = parse_state_string(state_str)

    required_fields = [
        "AngleToEnemy",
        "AngleToEnemyScore",
        "DistanceToEnemyScore",
        "NearBorderArenaScore",
        "FacingToArena"
    ]

    missing_fields = [f for f in required_fields if f not in state_dict]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    return query_action(
        angle=state_dict["AngleToEnemy"],
        angle_score=state_dict["AngleToEnemyScore"],
        dist_score=state_dict["DistanceToEnemyScore"],
        near_score=state_dict["NearBorderArenaScore"],
        facing=state_dict["FacingToArena"],
        top_k=top_k
    )

app = FastAPI(title="Sumobot Milvus Vector Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryInput(BaseModel):
    state: str
    top_k: Optional[int] = 1

class QueryResponse(BaseModel):
    raw_output: str
    action: Dict[str, Optional[float]]
    search_time_ms: float
    distance: float
    top_k_results: Optional[List[Dict]] = None

class BatchQueryInput(BaseModel):
    states: List[str]
    top_k: Optional[int] = 1

@app.get("/")
def root():
    return {
        "message": "Sumobot Milvus Vector Search API",
        "endpoints": {
            "health": "/health",
            "query": "/query (POST)",
            "batch": "/batch (POST)",
            "benchmark": "/benchmark (GET)",
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "collection": COLLECTION_NAME,
        "num_entities": col.num_entities,
        "platform": f"{platform.system()} {platform.machine()}",
        "mode": "GPU" if HAS_GPU else "CPU"
    }

@app.post("/query", response_model=QueryResponse)
def query(input: QueryInput):
    try:
        result = query_action_from_string(input.state, input.top_k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
def batch_query(input: BatchQueryInput):
    try:
        results = []
        total_time = 0

        for state in input.states:
            result = query_action_from_string(state, input.top_k)
            results.append(result)
            total_time += result["search_time_ms"]

        return {
            "results": results,
            "total_search_time_ms": round(total_time, 2),
            "avg_search_time_ms": round(total_time / len(results), 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/benchmark")
def benchmark():
    test_state = "AngleToEnemy=7.77, AngleToEnemyScore=0.99, DistanceToEnemyScore=0.76, NearBorderArenaScore=0.81, FacingToArena=-0.99"

    query_action_from_string(test_state)

    times = []
    num_runs = 100
    for _ in range(num_runs):
        result = query_action_from_string(test_state)
        times.append(result["search_time_ms"])

    times_sorted = sorted(times)

    return {
        "runs": num_runs,
        "stats": {
            "avg_latency_ms": round(sum(times) / len(times), 2),
            "min_latency_ms": round(min(times), 2),
            "max_latency_ms": round(max(times), 2),
            "p50_latency_ms": round(times_sorted[len(times) // 2], 2),
            "p95_latency_ms": round(times_sorted[int(len(times) * 0.95)], 2),
            "p99_latency_ms": round(times_sorted[int(len(times) * 0.99)], 2),
        },
        "platform": {
            "mode": "GPU" if HAS_GPU else "CPU",
            "collection_size": col.num_entities
        }
    }

if __name__ == "__main__":
    import sys
    import os

    port = int(os.getenv("PORT", "9999"))
    workers = int(os.getenv("WORKERS", "5"))

    if "--workers" in sys.argv:
        try:
            workers_idx = sys.argv.index("--workers")
            workers = int(sys.argv[workers_idx + 1])
        except (IndexError, ValueError):
            print(f"⚠️  Invalid --workers argument, using: {workers}")

    if "--port" in sys.argv:
        try:
            port_idx = sys.argv.index("--port")
            port = int(sys.argv[port_idx + 1])
        except (IndexError, ValueError):
            print(f"⚠️  Invalid --port argument, using: {port}")

    print(f"\n🚀 Starting server at http://0.0.0.0:{port}")
    print(f"👷 Workers: {workers}")
    print(f"📚 Docs: http://0.0.0.0:{port}/docs\n")

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        workers=workers,
        log_level="info"
    )