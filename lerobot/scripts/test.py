import torch
import json
import os
from safetensors import safe_open

# 체크포인트 경로 (사용하시는 경로로 수정)
model_path = "/home/jr/lerobot/outputs/train/so101_network_smolvlatrain/checkpoints/010000/pretrained_model"

print(f"📂 Checking model at: {model_path}")

# 1. config.json 확인
config_file = os.path.join(model_path, "config.json")
with open(config_file, 'r') as f:
    cfg = json.load(f)
    # output_features 혹은 policy.output_features 확인
    out_feat = cfg.get("output_features", cfg.get("policy", {}).get("output_features", {}))
    action_shape = out_feat.get("action", {}).get("shape", ["Unknown"])
    print(f"📄 config.json says Action Shape: {action_shape}")

# 2. 실제 가중치 파일 확인 (model.safetensors)
weight_file = os.path.join(model_path, "model.safetensors")
if os.path.exists(weight_file):
    with safe_open(weight_file, framework="pt") as f:
        # 모델마다 마지막 레이어 이름이 다를 수 있지만, 보통 action 관련 헤드임
        # SmolVLA/OpenVLA의 경우 action tokenizer나 linear layer를 확인
        keys = f.keys()
        action_keys = [k for k in keys if "action" in k or "head" in k or "linear" in k]
        
        print(f"\n🔍 Searching for output layers (Total keys: {len(keys)})...")
        for k in action_keys[-5:]: # 마지막 5개만 출력
            tensor = f.get_tensor(k)
            print(f"   - {k}: shape {tensor.shape}")

        print("\n💡 Tip: 만약 shape의 끝자리가 6 또는 7이라면 팔만 학습된 것이고,")
        print("        10이라면 바퀴까지 학습된 것입니다.")