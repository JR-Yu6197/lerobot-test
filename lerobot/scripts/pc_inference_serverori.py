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

# --- 유틸리티 함수 ---
def decode_image(img_bytes: bytes) -> np.ndarray:
    """바이트 데이터를 OpenCV 이미지 (BGR)로 복원합니다."""
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes.")
    return img

def load_ds_meta_from_policy_path(policy_path: str) -> Any:
    """
    정책 경로에서 config.json과 normalizer를 분석하여 ds_meta를 생성합니다.
    """
    # 1. config.json 로드
    config_path = os.path.join(policy_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
        
    with open(config_path, 'r') as f:
        policy_config = json.load(f)

    # 2. normalizer safetensors에서 stats 추출
    stats_dict = {}
    norm_file = "policy_preprocessor_step_5_normalizer_processor.safetensors"
    
    for file in os.listdir(policy_path):
        if "normalizer" in file and file.endswith(".safetensors"):
            norm_file = file
            break
            
    normalizer_path = os.path.join(policy_path, norm_file)

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
            print(f"✅ Loaded normalization stats from {norm_file}")
        except Exception as e:
             logging.warning(f"Failed to load stats from safetensors: {e}")
    else:
        logging.warning(f"⚠️ Normalizer file not found: {norm_file}. Stats might be incomplete.")

    # 3. Features 파싱
    if "input_features" not in policy_config:
        if "policy" in policy_config and "input_features" in policy_config["policy"]:
            policy_config = policy_config["policy"]
        else:
            raise KeyError("config.json missing 'input_features'. Check file format.")

    raw_features = {}
    raw_features.update(policy_config.get("input_features", {}))
    raw_features.update(policy_config.get("output_features", {}))

    converted_features = {}
    print("\n🔍 Processing Features Metadata:")
    
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
                print(f"  - {key}: Generated {dim} dummy names ({new_info['dtype']})")
            else:
                new_info["names"] = []
        else:
            print(f"  - {key}: Found existing names")

        converted_features[key] = new_info

    ds_meta = SimpleNamespace(
        features=converted_features,
        stats=stats_dict,
        fps=30
    )
    return ds_meta

# --- 메인 서버 클래스 ---
class InferenceServerController:
    def __init__(self, args):
        self.args = args
        self.device = get_safe_torch_device('cuda' if args.use_gpu else 'cpu')
        
        # 🌟 [핵심 수정] 강제로 Float32 사용 (BFloat16 충돌 방지)
        # Pi0 모델 내부에서 Noise 생성 시 float32를 사용하므로, 모델도 float32로 맞춰야 합니다.
        self.dtype = torch.float32 
        print(f"⚙️  Using device: {self.device}, dtype: {self.dtype} (Forced Float32)")

        # 모델 로드
        self.policy, self.preprocessor, self.postprocessor = self._load_policy_components(args.policy_path)
        
        # ZMQ 설정
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.bind_address = f"tcp://*:{args.port}"
        print(f"\n📡 Inference Server binding to {self.bind_address}...")
        self.socket.bind(self.bind_address)
        
        print("✅ Server Ready. Waiting for Robot Connection...")

    def _load_policy_components(self, policy_path: str):
        print(f"🔄 Loading Policy from {policy_path}...")
        
        ds_meta = load_ds_meta_from_policy_path(policy_path)
        
        conf = PreTrainedConfig.from_pretrained(policy_path)
        conf.pretrained_path = policy_path
        
        # [핵심] 컴파일 기능 끄기
        if hasattr(conf, "compile_model"):
            print("⚠️ Disabling compile_model to ensure inference stability.")
            conf.compile_model = False
        
        # Policy 생성
        policy = make_policy(conf, ds_meta=ds_meta) 
        
        # Processor 생성
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=conf,
            pretrained_path=conf.pretrained_path,
            dataset_stats=ds_meta.stats,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "rename_observations_processor": {"rename_map": {}}, 
            },
        )
        
        # GPU 이동 및 eval 모드
        try:
            policy.to(self.device, dtype=self.dtype)
        except TypeError:
            policy.to(self.device)
            
        policy.eval()
        
        print("✅ Policy loaded successfully.")
        return policy, preprocessor, postprocessor

    def _process_observation(self, header: Dict[str, Any], parts: List[bytes]) -> Dict[str, Any]:
        obs_dict = {}

        # State (dtype 맞춤)
        state_vec = torch.tensor(header["state"], dtype=self.dtype).to(self.device)
        obs_dict["observation.state"] = state_vec.unsqueeze(0) 

        # Images (dtype 맞춤)
        img_idx = 1
        for cam_key in header["image_keys"]:
            if img_idx >= len(parts): break
            
            img_bytes = parts[img_idx]
            img_bgr = decode_image(img_bytes)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.to(dtype=self.dtype) # Float32
            
            full_key = f"observation.images.{cam_key}"
            obs_dict[full_key] = img_tensor.unsqueeze(0).to(self.device)
            img_idx += 1
            
        # Task String
        obs_dict["task"] = [self.args.task] 

        return obs_dict

    def run(self):
        try:
            while True:
                # 1. 수신
                try:
                    parts = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue

                if not parts: continue

                # 2. 데이터 복원
                try:
                    header = json.loads(parts[0].decode('utf-8'))
                    obs_dict = self._process_observation(header, parts)
                except Exception as e:
                    logging.error(f"Data processing error: {e}")
                    self.socket.send_json({"error": str(e)})
                    continue

                # 3. 추론
                start = time.perf_counter()
                with torch.inference_mode():
                    processed_obs = self.preprocessor(obs_dict)
                    action_output = self.policy.select_action(processed_obs) 
                    action_vector = self.postprocessor(action_output)

                # 4. 전송
                action_dict = {}
                
                # Tensor 처리
                if isinstance(action_vector, torch.Tensor):
                    vals = action_vector.float().squeeze().cpu().numpy().tolist()
                    if not isinstance(vals, list): vals = [vals]
                    for i, val in enumerate(vals):
                        action_dict[f"action_{i}"] = val
                
                # Dict 처리
                elif isinstance(action_vector, dict):
                    for k, v in action_vector.items():
                        if isinstance(v, torch.Tensor):
                            val = v.float().squeeze().cpu().numpy().tolist()
                        else:
                            val = v
                        if isinstance(val, list) and len(val) == 1: val = val[0]
                        action_dict[k] = val

                self.socket.send_json(action_dict)
                
                dt = (time.perf_counter() - start) * 1000
                print(f"Inference: {dt:.1f}ms | Task: '{self.args.task}'", end='\r')

        except KeyboardInterrupt:
            print("\nServer Stopped.")
        finally:
            self.socket.close()
            self.context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--port", default=5555, type=int)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--task", default="Grab the cube", help="Instruction for VLM models")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    InferenceServerController(args).run()