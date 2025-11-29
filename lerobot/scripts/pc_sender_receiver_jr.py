import zmq
import json
import time
import argparse
import threading
import sys
import termios
import tty
import select
import numpy as np
import logging
import os
import signal
import cv2
from pathlib import Path
from datetime import datetime

# LeRobot
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PCSystem")

def get_raw_key():
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch1 = sys.stdin.read(1)
            if ch1 == '\x03': return "CTRL_C"
            return ch1.lower()
    except: pass
    return None

class PCSystem:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.keys = {'w': False, 's': False, 'a': False, 'd': False}
        self.speed = {"linear": 1.0, "angular": 1.5}
        self.saved_task = "Default task"
        self.ctx = zmq.Context()
        
        print(f"📡 PC Control Server (PUB) Binding on 0.0.0.0:6666")
        self.control_sock = self.ctx.socket(zmq.PUB)
        self.control_sock.bind(f"tcp://0.0.0.0:6666")

        print(f"📺 PC Video Server (PULL) Binding on 0.0.0.0:6667")
        self.video_sock = self.ctx.socket(zmq.PULL)
        self.video_sock.bind(f"tcp://0.0.0.0:6667")

        self.dataset = None
        self.last_arm_pos = {}
        for j in ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]:
            self.last_arm_pos[f"{j}.pos"] = 0.0

        # 리더암
        try:
            tid = args.teleop_id if args.teleop_id else "default_calibration_id"
            print(f"🦾 Connecting Leader (ID: {tid})...")
            conf = SO101LeaderConfig(port=args.teleop_port, id=tid)
            self.leader = SO101Leader(conf)
            self.leader.connect(calibrate=True)
            print("✅ Leader Connected!")
        except:
            print("⚠️ Leader Not Found (Using Keyboard Only)")
            self.leader = None

    def get_action_payload(self):
        if self.leader:
            try:
                obs = self.leader.get_action()
                if obs:
                    for k, v in obs.items():
                        keyname = k if k.endswith(".pos") else f"{k}.pos"
                        self.last_arm_pos[keyname] = float(v)
            except: pass

        action = self.last_arm_pos.copy()
        vx, vy = 0.0, 0.0
        if self.keys['w']: vx += self.speed["linear"]
        if self.keys['s']: vx -= self.speed["linear"]
        if self.keys['a']: vy += self.speed["angular"]
        if self.keys['d']: vy -= self.speed["angular"]
        
        m1 = vx - vy; m2 = vx - vy
        m3 = vx + vy; m4 = vx + vy
        
        action["base.motor_1"] = m1
        action["base.motor_2"] = m2
        action["base.motor_3"] = m3
        action["base.motor_4"] = m4
        action["base.linear_velocity"] = vx
        action["base.angular_velocity"] = vy
        return action

    def decode_frame(self, compressed_data):
        decoded = {}
        for k, v in compressed_data.items():
            if isinstance(v, bytes) and ("image" in k or k in ["wrist", "front"]):
                try:
                    nparr = np.frombuffer(v, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.transpose(img, (2, 0, 1)) / 255.0
                    decoded[k] = img.astype(np.float32)
                except: pass
            elif "_shape" in k: continue
            elif k == "task": decoded[k] = v
            else:
                try: decoded[k] = np.array([float(v)], dtype=np.float32)
                except: pass
        return decoded

    def video_loop(self):
        print("🎥 Video Thread Waiting...")
        frame_cnt = 0
        
        try:
            while self.running:
                try:
                    packet = self.video_sock.recv_pyobj(flags=zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001); continue
                except Exception: continue

                cmd = packet.get("command")
                if cmd == "STOP":
                    print("\n💾 [STOP RECEIVED] Saving...")
                    self.running = False
                    break
                
                raw = packet.get("data") or packet.get("frame")
                if not raw: continue
                
                obs = self.decode_frame(raw)
                if "task" not in obs: obs["task"] = self.saved_task

                # [핵심 수정] 데이터셋 생성 시 폴더 충돌 방지
                if self.dataset is None:
                    print("\n🎉 First Frame! Initializing...")
                    try:
                        features = self._infer_features(obs)
                        
                        # 시간 기반 유니크 이름 생성
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_name = self.args.repo_id.replace("/", "_")
                        unique_folder_name = f"{safe_name}_{timestamp}"
                        
                        # data_output 폴더 경로 지정 (미리 만들지 않음, LeRobot에 맡김)
                        output_root = Path("data_output") / unique_folder_name
                        
                        print(f"📁 Target Path: {output_root.absolute()}")

                        self.dataset = LeRobotDataset.create(
                            repo_id=self.args.repo_id,
                            root=output_root, 
                            fps=30,
                            features=features,
                            use_videos=True,
                            robot_type="so101_follower",
                            image_writer_processes=1 
                        )
                        self.encoding_manager = VideoEncodingManager(self.dataset)
                        self.encoding_manager.__enter__()
                        
                    except Exception as e: 
                        print(f"\n❌ DATASET INIT ERROR: {e}")
                        # 에러나면 저장 포기하고 주행이라도 하게 함
                        # self.running = False (주행을 위해 주석처리)
                        pass 

                if self.dataset:
                    self.dataset.add_frame(obs)
                    frame_cnt += 1
                    if frame_cnt % 30 == 0: print(f"REC: {frame_cnt} frames", end='\r')

        except Exception as e:
            print(f"Video Loop Error: {e}")
        
        finally:
            if self.dataset:
                print(f"\n💾 Finalizing... Saving to {self.dataset.root}")
                try:
                    self.dataset.save_episode()
                    self.encoding_manager.__exit__(None, None, None)
                    print(f"✅ SAVE COMPLETE!")
                except Exception as e:
                    print(f"⚠️ Save Failed: {e}")

    def _infer_features(self, frame):
        features = {}
        for k, v in frame.items():
            if k == "task": features[k] = {"dtype": "string", "shape": (1,), "names": None}
            elif isinstance(v, np.ndarray):
                if v.ndim == 3: features[k] = {"dtype": "video", "shape": v.shape, "names": ["c", "h", "w"]}
                else: features[k] = {"dtype": "float32", "shape": v.shape, "names": None}
        return features

    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

        t = threading.Thread(target=self.video_loop)
        t.start()

        print("\n⌨️  READY! Press 'q' to quit.")

        try:
            while self.running:
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1).lower()
                    if ch == 'q':
                        print("\n🛑 Stopping...")
                        self.running = False 
                        for _ in range(5):
                            self.control_sock.send_string(json.dumps({"event": "stop_recording"}))
                            time.sleep(0.1)
                        break
                    
                    if ch in ['w', 'a', 's', 'd']: self.keys[ch] = True
                    if ch == 'n': self.control_sock.send_string(json.dumps({"event": "next_episode"}))

                msg = self.get_action_payload()
                self.control_sock.send_string(json.dumps(msg))

                vx = msg.get('base.linear_velocity', 0)
                if vx != 0:
                    print(f"\r🚀 MOVE: {vx:.1f}   ", end="")
                else:
                    print(f"\r💤 IDLE      ", end="")
                
                if not select.select([sys.stdin], [], [], 0)[0]:
                     self.keys = {x: False for x in self.keys}

                time.sleep(0.05)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.running = False
            t.join(timeout=5)
            self.ctx.term()
            print("👋 Bye.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo.id", dest="repo_id", required=True)
    parser.add_argument("--video.port", dest="video_port", default=6667, type=int)
    parser.add_argument("--teleop.port", dest="teleop_port", default="/dev/ttyACM0")
    parser.add_argument("--teleop.id", dest="teleop_id", default=None)
    args = parser.parse_args()
    PCSystem(args).run()