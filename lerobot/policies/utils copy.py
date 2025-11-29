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
    정책 경로에서 config.json과 normalizer safetensors를 분석하여
    make_policy에 필요한 ds_meta 객체(SimpleNamespace)를 구성합니다.
    """
    
    # 1. config.json 로드
    config_path = os.path.join(policy_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {config_path}")
        
    with open(config_path, 'r') as f:
        policy_config = json.load(f)

    # 2. normalizer safetensors에서 stats 추출
    normalizer_path = os.path.join(policy_path, "policy_preprocessor_step_5_normalizer_processor.safetensors")
    stats_dict = {}
    if os.path.exists(normalizer_path):
        try:
            with safe_open(normalizer_path, framework="pt") as f:
                for key in f.keys():
                    # Key format: obs_key/mean, obs_key/std
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

    # 3. ds_meta 객체 구성 및 [강제 보정]
    if "input_features" not in policy_config or "output_features" not in policy_config:
        raise KeyError("config.json does not contain 'input_features' or 'output_features'.")

    # Raw Features 병합
    raw_features = {}
    raw_features.update(policy_config["input_features"])
    raw_features.update(policy_config["output_features"])

    # 🌟 [핵심 수정] 이름(names) 및 타입(dtype) 강제 주입
    converted_features = {}
    for key, feat_info in raw_features.items():
        new_info = feat_info.copy()
        feat_type = feat_info.get("type")
        
        # (1) Dtype 보정
        if feat_type == "VISUAL":
            new_info["dtype"] = "video" 
        else:
            new_info["dtype"] = "float32"

        # (2) Names 강제 생성 (STATE, ACTION이고 names가 없을 때)
        if feat_type in ["STATE", "ACTION"]:
            if "names" not in new_info:
                # shape 정보 확인 (예: [10])
                shape = new_info.get("shape", [])
                if shape:
                    dim = shape[0]
                    # 가짜 이름 생성 (예: action_0, action_1...)
                    # 주의: 로봇 PC에서 이 순서대로 모터에 매핑해야 함
                    new_info["names"] = [f"{key}_{i}" for i in range(dim)]
                    print(f"⚠️ Generated dummy names for {key}: {new_info['names']}")
                else:
                    logging.warning(f"Feature {key} has no shape information!")

        converted_features[key] = new_info

    # SimpleNamespace로 객체화
    ds_meta = SimpleNamespace(
        features=converted_features,
        stats=stats_dict,
        fps=30
    )
    
    return ds_meta

# --- 메인 컨트롤러 클래스 ---
class InferenceServerController:
    def __init__(self, args):
        self.args = args
        self.device = get_safe_torch_device('cuda' if args.use_gpu else 'cpu')
        
        # --- 1. 모델 및 프로세서 로드 ---
        self.policy, self.preprocessor, self.postprocessor = self._load_policy_components(args.policy_path)
        
        # --- 2. ZMQ 설정 (REP 모드) ---
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.bind_address = f"tcp://*:{args.port}"
        print(f"📡 Inference Server binding to {self.bind_address} on {self.device}...")
        self.socket.bind(self.bind_address)
        
        print("✅ Server Ready. Waiting for Robot Connection...")

    def _load_policy_components(self, policy_path: str) -> Tuple[Any, PolicyProcessorPipeline, PolicyProcessorPipeline]:
        print(f"🔄 Loading Policy components from {policy_path}...")
        
        # 1. ds_meta 객체 구성 (Names 자동 생성 포함)
        ds_meta = load_ds_meta_from_policy_path(policy_path)
        dataset_stats = ds_meta.stats
        
        # 2. 정책 설정 로드
        conf = PreTrainedConfig.from_pretrained(policy_path)
        conf.pretrained_path = policy_path
        
        # 3. 정책 생성 (이제 names 에러가 나지 않아야 함)
        policy = make_policy(conf, ds_meta=ds_meta) 

        # 4. 전처리/후처리 생성
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=conf,
            pretrained_path=conf.pretrained_path,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
                "rename_observations_processor": {"rename_map": {}}, 
            },
        )
        
        # 5. GPU 이동 및 eval 모드
        policy.to(self.device).eval()
        preprocessor.to(self.device)
        postprocessor.to(self.device)
        
        print("✅ Policy components loaded successfully.")
        return policy, preprocessor, postprocessor

    def _process_observation(self, header: Dict[str, Any], parts: List[bytes]) -> Dict[str, Any]:
        """ZMQ 멀티파트 데이터를 LeRobot Observation Dict로 변환"""
        obs_dict = {}

        # 1. State (모터 값)
        state_vec = torch.tensor(header["state"], dtype=torch.float32).to(self.device)
        obs_dict["observation.state"] = state_vec.unsqueeze(0) 

        # 2. Images (카메라)
        img_idx = 1
        for cam_key in header["image_keys"]:
            if img_idx >= len(parts):
                logging.warning(f"Missing image data for key: {cam_key}")
                continue
                
            img_bytes = parts[img_idx]
            img_bgr = decode_image(img_bytes)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            
            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            
            full_key = f"observation.images.{cam_key}"
            obs_dict[full_key] = img_tensor.unsqueeze(0).to(self.device)
            
            img_idx += 1
            
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
                except (json.JSONDecodeError, ValueError) as e:
                    logging.error(f"Data processing error: {e}")
                    self.socket.send_json({"error": f"Data decode failed: {e}"})
                    continue

                # 3. 추론
                start = time.perf_counter()
                with torch.no_grad():
                    processed_obs = self.preprocessor(obs_dict)
                    action_output = self.policy.select_action(processed_obs) 
                    action_vector = self.postprocessor(action_output)

                # 4. 전송 (Dict 변환)
                action_dict = {}
                for k, v in action_vector.items():
                    # 텐서 처리: (1, Dim) -> List
                    if v.dim() > 0:
                        action_dict[k] = v.squeeze(0).cpu().numpy().tolist()
                    else:
                        action_dict[k] = v.item()

                self.socket.send_json(action_dict)
                
                inference_time_ms = (time.perf_counter() - start) * 1000
                print(f"Inference done in {inference_time_ms:.1f}ms | Sent action.", end='\r')

        except KeyboardInterrupt:
            print("\nServer Stopped.")
        except Exception as e:
            logging.error(f"Critical error: {e}")
        finally:
            self.socket.close()
            self.context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy.path", dest="policy_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--port", default=5555, type=int)
    parser.add_argument("--use_gpu", action="store_true")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    InferenceServerController(args).run()