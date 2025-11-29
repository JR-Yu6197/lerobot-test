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
from pathlib import Path
from datetime import datetime

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.datasets.video_utils import VideoEncodingManager
    from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig
except ImportError:
    print("⚠️ LeRobot 라이브러리 없음 (Dummy Mode)")
    LeRobotDataset = None
    SO101Leader = None

# ==============================================================================
# 🕵️‍♂️ [DEBUG] PC용 상세 로거
# ==============================================================================
class DebugFormatter(logging.Formatter):
    def format(self, record):
        timestamp = time.strftime('%H:%M:%S', time.localtime(record.created))
        msecs = int(record.msecs)
        return f"[{timestamp}.{msecs:03d}] {record.getMessage()}"

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(DebugFormatter())
logger = logging.getLogger("PCDebug")
logger.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.propagate = False

# ==============================================================================
# ⌨️ 키보드 유틸
# ==============================================================================
def get_raw_key():
    try:
        if select.select([sys.stdin], [], [], 0)[0]:
            ch1 = sys.stdin.read(1)
            if ch1 == '\x03': return "CTRL_C"
            if ch1 == '\x1b':
                seq = sys.stdin.read(2)
                if seq == '[A': return 'w'
                if seq == '[B': return 's'
                if seq == '[C': return 'd'
                if seq == '[D': return 'a'
            return ch1.lower()
    except Exception: pass
    return None

# ==============================================================================
# 🖥️ PC 시스템
# ==============================================================================
class PCSystem:
    def __init__(self, args):
        self.args = args
        self.running = True
        self.keys = {'w': False, 's': False, 'a': False, 'd': False}
        self.speed = {"linear": 0.4, "angular": 0.8}
        self.saved_task = "Default teleoperation task"

        self.ctx = zmq.Context()
        
        # 1. Control (REQ)
        logger.info(f"📡 [ZMQ] Control 연결 시도: {args.rpi_ip}:{args.rpi_port}")
        self.control_sock = self.ctx.socket(zmq.REQ)
        self.control_sock.connect(f"tcp://{args.rpi_ip}:{args.rpi_port}")
        self.control_sock.setsockopt(zmq.RCVTIMEO, 2000)
        self.control_sock.setsockopt(zmq.LINGER, 0)

        # 2. Video (PULL)
        logger.info(f"📺 [ZMQ] Video 서버 Bind: 0.0.0.0:{args.video_port}")
        self.video_sock = self.ctx.socket(zmq.PULL)
        self.video_sock.bind(f"tcp://0.0.0.0:{args.video_port}")
        self.video_sock.setsockopt(zmq.RCVHWM, 1)

        self.dataset = None
        self.encoding_manager = None
        self.last_arm_pos = {k: 0.0 for k in ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos", "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]}
        self.leader = None
        
        if SO101Leader and args.teleop_id:
            logger.info(f"🦾 [Leader] 연결 시도 ID: {args.teleop_id}")
            # (생략: 리더 연결 로직은 위와 동일)

    def get_action_payload(self):
        action = self.last_arm_pos.copy()
        vx, vy = 0.0, 0.0
        if self.keys['w']: vx += self.speed["linear"]
        if self.keys['s']: vx -= self.speed["linear"]
        if self.keys['a']: vy += self.speed["angular"]
        if self.keys['d']: vy -= self.speed["angular"]
        action["base.linear_velocity"] = vx
        action["base.angular_velocity"] = vy
        return action

    def video_loop(self):
        logger.info("🎥 [Video Thread] 시작 - 데이터 대기 중...")
        frame_cnt = 0
        last_log_time = time.time()
        
        while self.running:
            try:
                # 1. Receive
                try:
                    packet = self.video_sock.recv_pyobj(flags=zmq.NOBLOCK)
                    # logger.debug("  📦 [Packet] 도착") # 패킷 도착 즉시 확인
                except zmq.Again:
                    # 3초간 데이터가 없으면 경고
                    if time.time() - last_log_time > 3.0:
                        logger.warning("  ⚠️ [No Data] 3초째 영상 데이터 없음 (네트워크/로봇 확인)")
                        last_log_time = time.time()
                    time.sleep(0.005)
                    continue
                except Exception as e:
                    logger.error(f"❌ [Video Error] {e}")
                    continue

                cmd = packet.get("command")
                
                # 2. INIT Log
                if cmd == "INIT":
                    logger.info("  ✨ [INIT] 초기화 패킷 수신")
                    continue

                # 3. FRAME Processing
                if cmd == "FRAME":
                    frame = packet.get("frame")
                    ts = packet.get("timestamp", 0)
                    latency = (time.time() - ts) * 1000
                    
                    # 지연 시간 확인
                    if latency > 300:
                        logger.warning(f"  🐢 [Lag] 지연 심각: {latency:.1f}ms")
                    
                    # 최초 데이터셋 생성
                    if self.dataset is None and LeRobotDataset:
                        logger.info("  📁 [Dataset] 생성 시도...")
                        # (생략: 데이터셋 생성 로직은 이전과 동일)
                        # 성공 시:
                        logger.info("  ✅ [Dataset] 생성 완료")

                    if self.dataset:
                        self.dataset.add_frame(frame)
                        frame_cnt += 1
                        if frame_cnt % 30 == 0:
                            print(f"  🔴 [REC] Frames: {frame_cnt} | Latency: {latency:.1f}ms", end='\r')
                    continue
                
                if cmd == "STOP":
                    logger.info("  💾 [STOP] 저장 명령 수신")
                    if self.dataset: self.dataset.save_episode()

            except Exception as e:
                logger.error(f"❌ [Critical] Video Loop: {e}")
                time.sleep(1)

    def run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        
        t = threading.Thread(target=self.video_loop, daemon=True)
        t.start()
        
        logger.info("🚀 [Main] PC Control Start")
        
        try:
            while self.running:
                k = get_raw_key()
                if k == 'q' or k == 'CTRL_C':
                    self.running = False; break
                
                if k:
                    # logger.debug(f"  ⌨️ [Key] 입력: {k}") # 키 입력 확인
                    pass
                
                if k in ['w', 'a', 's', 'd']: self.keys[k] = True
                else: self.keys = {x: False for x in self.keys}
                
                msg = self.get_action_payload()
                if k == 'n': msg = {"event": "next_episode"}
                
                try:
                    # logger.debug(f"  📤 [Send] {msg}") # 보내는 메시지 확인
                    self.control_sock.send_string(json.dumps(msg))
                    self.control_sock.recv_string() # ACK
                except zmq.Again:
                    logger.warning("  ⚠️ [Timeout] 로봇 응답 없음 (재연결 시도)")
                    self.control_sock.close()
                    self.control_sock = self.ctx.socket(zmq.REQ)
                    self.control_sock.connect(f"tcp://{self.args.rpi_ip}:{self.args.rpi_port}")
                    self.control_sock.setsockopt(zmq.RCVTIMEO, 2000)
                    self.control_sock.setsockopt(zmq.LINGER, 0)
                
                time.sleep(0.05)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            self.running = False
            self.ctx.term()
            logger.info("👋 [Bye] 종료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpi.ip", dest="rpi_ip", required=True)
    parser.add_argument("--rpi.port", dest="rpi_port", default=5555, type=int)
    parser.add_argument("--video.port", dest="video_port", default=5556, type=int)
    parser.add_argument("--repo.id", dest="repo_id", default="debug_session")
    parser.add_argument("--teleop.port", dest="teleop_port", default="/dev/ttyACM0")
    parser.add_argument("--teleop.id", dest="teleop_id", default=None)
    args = parser.parse_args()
    PCSystem(args).run()