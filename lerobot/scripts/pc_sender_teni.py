import zmq
import json
import time
import argparse
import threading
import sys
import termios  # 키보드 입력
import tty      # 키보드 입력

# LeRobot 라이브러리
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

# --- SSH/Terminal 기반 키보드 리스너 (WASD 안정화) ---
class KeyboardListener:
    def __init__(self, on_press, on_release):
        self.on_press = on_press
        self.on_release = on_release
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.5)

    def _getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def _run(self):
        while self.running:
            try:
                key = self._getch()
                if key == '\x03': self.running = False; break
                self.on_press(key)
                time.sleep(0.05)
                self.on_release(key)
            except Exception: break
# ----------------------------------------------------

class PCRemoteController:
    def __init__(self, args):
        self.args = args
        self.key_states = {'w': False, 's': False, 'a': False, 'd': False}
        self.speed_settings = {'linear_max': 0.4, 'angular_max': 0.8}
        
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVTIMEO, 2000)
        self.socket.setsockopt(zmq.LINGER, 0)
        
        print(f"📡 Connecting to Pi at {args.rpi_ip}:{args.rpi_port}...")
        self.socket.connect(f"tcp://{args.rpi_ip}:{args.rpi_port}")

        self.arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.last_arm_pos = {f"{j}.pos": 0.0 for j in self.arm_joints} # 항상 6개 키 유지

        # 리더 암 설정
        try:
            print(f"🦾 Connecting to Leader Arm at {args.teleop_port}...")
            conf = SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id)
            self.leader = SO101Leader(conf)

            self.leader.connect(calibrate=True)
            print("✅ Leader Arm Connected!")

            # 캘리브레이션 디버깅 확인 코드
            print(f"DEBUG: Is Calibrated? {self.leader.is_calibrated}")
            print(f"DEBUG: Calibration File Path: {self.leader.calibration_fpath}")

        except Exception as e:
            print(f"❌ Leader Arm Error: {e}")
            self.leader = None

        self.listener = KeyboardListener(on_press=self._on, on_release=self._off)
        self.listener.start()

    def _on(self, key):
        k = key.lower()
        if k in self.key_states: self.key_states[k] = True
    def _off(self, key):
        k = key.lower()
        if k in self.key_states: self.key_states[k] = False

    def get_combined_action(self):
        action = self.last_arm_pos.copy() 
        
        # 1. 리더 암 데이터 읽기
        if self.leader and self.leader.is_connected:
            try:
                # 🌟🌟🌟 FIX: get_observation() 대신 get_action() 사용 🌟🌟🌟
                obs = self.leader.get_action()  # Leader Arm의 명령 데이터를 가져옵니다.
                # obs = self.leader.get_observation()
                if obs:
                    # 읽어온 Raw Arm 데이터를 콘솔에 출력합니다.
                    print(f"DEBUG RAW ARM: {obs}", end='\r')
                
                # 데이터가 있을 경우에만 갱신 (나머지 수동 매핑 로직 유지)
                if obs:
                    for k, v in obs.items():
                        key = k if k.endswith(".pos") else f"{k}.pos"
                        if key in action:
                             action[key] = float(v)
                        
            except Exception as e:
                # 데이터 읽기 실패 시 경고
                pass
            
        # 2. 키보드 데이터 (속도)
        vx, vyaw = 0.0, 0.0
        if self.key_states['w']: vx += self.speed_settings['linear_max']
        if self.key_states['s']: vx -= self.speed_settings['linear_max']
        if self.key_states['a']: vyaw += self.speed_settings['angular_max']
        if self.key_states['d']: vyaw -= self.speed_settings['angular_max']
        
        action["base.linear_velocity"] = float(vx)
        action["base.angular_velocity"] = float(vyaw)
        
        return action

    def run(self):
        print("\n🚀 Running! Use WASD to move the car. Press Ctrl+C to exit.")
        
        try:
            while True:
                start = time.perf_counter()
                action = self.get_combined_action()
                
                self.socket.send_string(json.dumps(action))
                
                # ACK 대기 (재연결 로직)
                try:
                    _ = self.socket.recv_string()
                    arm_keys = sum(1 for k in action if ".pos" in k)
                    print(f"Sent: Vel={action['base.linear_velocity']:.1f} | ArmKeys={arm_keys}", end='\r')
                except zmq.Again:
                    # Timeout 발생 시 재연결
                    self.socket.close()
                    self.socket = self.context.socket(zmq.REQ)
                    self.socket.setsockopt(zmq.RCVTIMEO, 2000)
                    self.socket.connect(f"tcp://{self.args.rpi_ip}:{self.args.rpi_port}")
                    print("⚠️ Reconnecting...", end='\r')
                    continue

                time.sleep(max(0, 0.033 - (time.perf_counter() - start))) # 약 30Hz

            
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            if self.leader: self.leader.disconnect()
            self.listener.stop()
            self.socket.close()
            self.context.term()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpi.ip", dest="rpi_ip", required=True)
    parser.add_argument("--teleop.port", dest="teleop_port", default="/dev/ttyACM0")
    parser.add_argument("--rpi.port", dest="rpi_port", default=5555, type=int)

    # 리더암 ID
    parser.add_argument("--teleop.id", dest="teleop_id", default=None)

    PCRemoteController(parser.parse_args()).run()