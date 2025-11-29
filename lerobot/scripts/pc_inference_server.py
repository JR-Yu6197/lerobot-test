import zmq
import torch
import cv2
import numpy as np
import json
import time
import argparse
import logging
import os
from typing import Dict, Any, List, Tuple
from types import SimpleNamespace

# LeRobot Policies and Utilities
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.utils.utils import get_safe_torch_device
from safetensors import safe_open

def decode_image(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    return img

def load_ds_meta_from_policy_path(policy_path: str) -> Any:
    config_path = os.path.join(policy_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
        
    with open(config_path, 'r') as f:
        policy_config = json.load(f)

    normalizer_path = os.path.join(policy_path, "policy_preprocessor_step_5_normalizer_processor.safetensors")
    stats_dict = {}
    if os.path.exists(normalizer_path):
        try:
            with safe_open(normalizer_path, framework="pt") as f:
                for key in f.keys():
                    parts = key.split('/')
                    if len(parts) == 2:
                        name, stat_type = parts
                        tensor = f.get_tensor(key)
                        value = tensor.cpu().numpy().tolist()
                        if name not in stats_dict: stats_dict[name] = {}
                        stats_dict[name][stat_type] = value
            print("✅ Loaded normalization stats from safetensors.")
        except Exception as e:
             logging.warning(f"Failed to load stats from safetensors: {e}. Normalization may fail.")
    else:
        logging.warning(f"⚠️ Normalizer safetensors not found at {normalizer_path}. Stats might be incomplete.")

    if "input_features" not in policy_config or "output_features" not in policy_config:
        raise KeyError("config.json does not contain 'input_features' or 'output_features'.")

    raw_features = {}
    raw_features.update(policy_config["input_features"])
    raw_features.update(policy_config["output_features"])

    converted_features = {}
    print("\n🔍 Processing Features Metadata (Defensive Mode):")
    
    for key, feat_info in raw_features.items():
        new_info = feat_info.copy()
        feat_type = str(feat_info.get("type", "")).strip().upper()
        
        if feat_type == "VISUAL" or "image" in key.lower():
            new_info["dtype"] = "video"
        else:
            new_info["dtype"] = "float32"

        if "names" not in new_info:
            shape = new_info.get("shape", [])
            if shape:
                dim = shape[0]
                new_info["names"] = [f"{key}_{i}" for i in range(dim)]
                print(f"  - {key} ({new_info['dtype']}): Generated {dim} dummy names")
            else:
                new_info["names"] = []
                print(f"  - {key}: No shape info, assigned empty names.")
        else:
            print(f"  - {key}: Found existing names")

        converted_features[key] = new_info

    ds_meta = SimpleNamespace(
        features=converted_features,
        stats=stats_dict,
        fps=30
    )
    return ds_meta

class InferenceServerController:
    def __init__(self, args):
        self.args = args
        self.device = get_safe_torch_device('cuda' if args.use_gpu else 'cpu')
        self.current_task = args.task
        
        self.policy, self.preprocessor, self.postprocessor = self._load_policy_components(args.policy_path)
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.bind_address = f"tcp://*:{args.port}"
        print(f"\n📡 Inference Server binding to {self.bind_address} on {self.device}...")
        self.socket.bind(self.bind_address)
        
        print("✅ Server Ready. Waiting for Robot Connection...")

    def set_task(self, task: str):
        print(f"[TASK] Updated task: {task}")
        self.current_task = task

    def _load_policy_components(self, policy_path: str) -> Tuple[Any, PolicyProcessorPipeline, PolicyProcessorPipeline]:
        print(f"🔄 Loading Policy components from {policy_path}...")
        ds_meta = load_ds_meta_from_policy_path(policy_path)
        dataset_stats = ds_meta.stats
        
        conf = PreTrainedConfig.from_pretrained(policy_path)
        conf.pretrained_path = policy_path
        
        policy = make_policy(conf, ds_meta=ds_meta) 
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=conf,
            pretrained_path=conf.pretrained_path,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "rename_observations_processor": {"rename_map": {}}, 
            },
        )
        policy.to(self.device).eval()
        print("✅ Policy components loaded successfully.")
        return policy, preprocessor, postprocessor

    def _process_observation(self, header: Dict[str, Any], parts: List[bytes]) -> Dict[str, Any]:
        """ZMQ 멀티파트 데이터를 LeRobot Observation Dict로 변환"""
        
        if "state" not in header:
            if "hello" in header:
                raise ValueError("Handshake packet received in inference loop. Please reset client.")
            raise ValueError(f"Missing 'state' key in header. Keys found: {list(header.keys())}")

        obs_dict = {}

        # 1. State
        state_data = header["state"]
        if not isinstance(state_data, list):
            raise ValueError(f"'state' must be a list, got {type(state_data)}")
            
        state_vec = torch.tensor(state_data, dtype=torch.float32).to(self.device)
        obs_dict["observation.state"] = state_vec.unsqueeze(0) 

        # 2. Images
        img_idx = 1
        for cam_key in header.get("image_keys", []):
            if img_idx >= len(parts):
                logging.warning(f"Missing image data for key: {cam_key}")
                continue
                
            img_bytes = parts[img_idx]
            if len(img_bytes) == 0:
                 # logging.warning(f"Empty image bytes for {cam_key}")
                 img_tensor = torch.zeros((3, 480, 640), dtype=torch.float32)
            else:
                img_bgr = decode_image(img_bytes)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
            full_key = f"observation.images.{cam_key}"
            obs_dict[full_key] = img_tensor.unsqueeze(0).to(self.device)
            img_idx += 1
            
        obs_dict["task"] = [self.current_task] 
        return obs_dict

    def run(self):
        # 1) Handshake
        print("⏳ Waiting for handshake from robot client...")
        try:
            while True:
                if self.socket.poll(100):
                    msg = self.socket.recv_json()
                    if "hello" in msg:
                        print("🤝 Handshake received:", msg)
                        self.socket.send_json({"ready": True})
                        print("🚀 Handshake completed. Starting inference...")
                        break
                    else:
                        print(f"⚠️ Unexpected handshake msg: {msg}")
                        self.socket.send_json({"error": "waiting_for_hello"})
        except KeyboardInterrupt:
            print("\nServer Stopped during handshake.")
            return
        except Exception as e:
            print(f"❌ Handshake Error: {e}")
            return

        # 2) Inference Loop
        print("✅ Inference loop started.")
        while True:
            try:
                # 1. 수신 (블로킹)
                parts = self.socket.recv_multipart()
                if not parts: continue

                start = time.perf_counter()

                # 2. 데이터 복원 및 검증
                try:
                    header = json.loads(parts[0].decode('utf-8'))
                    obs_dict = self._process_observation(header, parts)
                except Exception as e:
                    error_msg = f"Data processing error: {str(e)}"
                    logging.error(error_msg)
                    self.socket.send_json({"error": error_msg})
                    continue

                # =========================================================
                # 🔥 [수정됨] Task 상태별 동작 로직 (Pause / Stop / Inference)
                # =========================================================
                
                task_cmd = self.current_task.lower()
                action_dict = {}

                # [상황 1] 기본 대기/일시 정지 (Pause/Hold/Idle) 
                # -> "움직이지 말고 현재 자세 유지해" (기본값 Idle을 여기로 포함)
                if task_cmd in ["pause", "hold", "idle"]:
                    action_dict["is_pause_signal"] = True
                
                # [상황 2] 강제 정지/홈 포지션 (Stop/Home) 
                # -> "안전한 'ㄱ'자 자세로 이동해" (명시적으로 'stop'이나 'home'을 줬을 때만)
                elif task_cmd in ["stop", "home", "reset", "wait"]:
                    # 안전 대기 자세 (ㄱ자)
                    safe_idle_pose = [-0.45, -44.19, 59.54, 21.15, -56.03, 13.98, 0.0, 0.0, 0.0, 0.0]
                    
                    for i, val in enumerate(safe_idle_pose):
                        action_dict[f"action_{i}"] = val
                    
                    # Stop/Home 명령일 때는 is_pause_signal을 보내지 않음 -> 로봇이 이동해야 함

                # [상황 3] 모델 추론 (Inference) -> AI가 움직임 제어
                else:
                    try:
                        with torch.no_grad():
                            processed_obs = self.preprocessor(obs_dict)
                            action_output = self.policy.select_action(processed_obs) 
                            action_vector = self.postprocessor(action_output)

                        # 액션 결과 파싱
                        if isinstance(action_vector, torch.Tensor):
                            vals = action_vector.squeeze().cpu().numpy().tolist()
                        elif isinstance(action_vector, dict):
                            flatten = []
                            for v in action_vector.values():
                                if isinstance(v, torch.Tensor):
                                    flatten.extend(v.squeeze().cpu().numpy().tolist())
                                elif isinstance(v, list):
                                    flatten.extend(v)
                                else:
                                    flatten.append(v)
                            vals = flatten
                        else:
                            raise ValueError(f"Unsupported action type: {type(action_vector)}")

                        vals = vals[:10]
                        action_dict = {f"action_{i}": float(vals[i]) for i in range(10)}

                    except Exception as e:
                        logging.error(f"[ACTION_PARSE_FAIL] {e}")
                        self.socket.send_json({"error": "action_parse_fail"})
                        continue
                
                # =========================================================
                # 5. 전송
                self.socket.send_json(action_dict)
                
                inference_time_ms = (time.perf_counter() - start) * 1000
                
                # 로그 출력 정리
                if task_cmd in ["pause", "hold", "idle"]:
                     print(f"⏸️  PAUSED ({inference_time_ms:.1f}ms) - Holding Position", end='\r')
                elif task_cmd in ["stop", "home", "wait"]:
                     print(f"🛑 STOPPED ({inference_time_ms:.1f}ms) - Moving to Home", end='\r')
                else:
                     print(f"⚡ Inference ({inference_time_ms:.1f}ms) | Task: {self.current_task}", end='\r')

            except KeyboardInterrupt:
                print("\n🛑 Server Stopped.")
                break
            except Exception as e:
                logging.error(f"Critical loop error: {e}")
                try: self.socket.send_json({"error": "critical_server_error"})
                except: pass

        self.socket.close()
        self.context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--port", default=5555, type=int)
    parser.add_argument("--use_gpu", action="store_true")
    # 시작 시 기본 태스크를 'Idle'(제자리 유지)로 설정
    parser.add_argument("--task", default="Idle") 
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    InferenceServerController(args).run()