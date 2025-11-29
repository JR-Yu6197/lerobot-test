import zmq
import json
import time
import argparse
import threading
import sys
import termios
import tty
import select

# ================================================================
# RAW 방향키 조합기
# ================================================================
def get_raw_key():
    """
    방향키와 ESC의 분리 입력을 정확히 조합하는 함수
    ← = \x1b[D
    → = \x1b[C
    ESC 단독 = \x1b
    """
    ch1 = sys.stdin.read(1)

    # Ctrl + C
    if ch1 == '\x03':
        return "CTRL_C"

    # ESC 관련
    if ch1 == '\x1b':
        seq = ch1

        # ESC 이후 최대 2글자까지 읽어본다
        for _ in range(2):
            r, _, _ = select.select([sys.stdin], [], [], 0.03)
            if r:
                seq += sys.stdin.read(1)
            else:
                break

        # 단독 ESC
        if seq == '\x1b':
            return "ESC"

        # ←
        if seq == '\x1b[D':
            return "LEFT"

        # →
        if seq == '\x1b[C':
            return "RIGHT"

        # 기타 ESC 조합키는 무시
        return None

    # 일반 문자키
    return ch1


# ================================================================
# RAW Keyboard Listener
# ================================================================
class RawKeyboardListener:
    def __init__(self, on_press, on_release, on_ctrl_c=None):
        self.on_press = on_press
        self.on_release = on_release
        self.on_ctrl_c = on_ctrl_c
        self.running = False
        self.thread = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=0.2)

    def _run(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setraw(fd)

            while self.running:
                # 키 입력 여부 감지
                rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
                if not rlist:
                    continue

                key = get_raw_key()

                # Ctrl+C 처리
                if key == "CTRL_C":
                    if self.on_ctrl_c:
                        self.on_ctrl_c()
                    break

                # 방향키 / ESC 처리
                if key in ("ESC", "LEFT", "RIGHT"):
                    self.on_press(key)
                    time.sleep(0.05)
                    self.on_release(key)
                    continue

                # 일반키
                if isinstance(key, str) and len(key) == 1:
                    k = key.lower()
                    self.on_press(k)
                    time.sleep(0.05)
                    self.on_release(k)

        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ================================================================
# PCUnifiedSender (기존 로봇 제어기)
# ================================================================
from lerobot.teleoperators.so101_leader import SO101Leader, SO101LeaderConfig

class PCUnifiedSender:
    def __init__(self, args):
        # 이동키
        self.key_states = {"w": False, "s": False, "a": False, "d": False}
        self.speed = {"linear": 0.4, "angular": 0.8}

        # 이벤트 저장
        self.pending_event = None
        self.ctrl_c_pressed = False

        # ZMQ
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://{args.rpi_ip}:{args.rpi_port}")
        print(f"📡 Connected to RPi at {args.rpi_ip}:{args.rpi_port}")

        # 리더암
        self.arm_joints = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                           "wrist_flex", "wrist_roll", "gripper"]
        self.last_arm_pos = {f"{j}.pos": 0.0 for j in self.arm_joints}

        try:
            conf = SO101LeaderConfig(port=args.teleop_port, id=args.teleop_id)
            self.leader = SO101Leader(conf)
            self.leader.connect(calibrate=True)
            print("🦾 Leader Arm Connected!")
        except Exception as e:
            print("❌ Leader Arm Error:", e)
            self.leader = None

        # RAW listener 시작
        self.listener = RawKeyboardListener(
            on_press=self._on_key_press,
            on_release=self._on_key_release,
            on_ctrl_c=self._on_ctrl_c
        )
        self.listener.start()

    # 키 눌림 처리
    def _on_key_press(self, key):
        if key in self.key_states:
            self.key_states[key] = True
            return

        if key == "q":
            self.pending_event = {"event": "stop_recording"}
            print("\n📤 EVENT: stop_recording")

        elif key == "n":  # NEXT EPISODE
            self.pending_event = {"event": "next_episode"}
            print("\n📤 EVENT: next_episode")

        elif key == "z":  # RERECORD
            self.pending_event = {"event": "rerecord_episode"}
            print("\n📤 EVENT: rerecord_episode")
            
    # 키 떼기 처리
    def _on_key_release(self, key):
        if key in self.key_states:
            self.key_states[key] = False

    def _on_ctrl_c(self):
        print("\n🛑 Ctrl+C pressed")
        self.ctrl_c_pressed = True

    # 로봇 액션 빌드
    def build_action(self):
        action = self.last_arm_pos.copy()

        # 리더암
        if self.leader and self.leader.is_connected:
            try:
                obs = self.leader.get_action()
                if obs:
                    for k, v in obs.items():
                        keyname = k if k.endswith(".pos") else f"{k}.pos"
                        if keyname in action:
                            action[keyname] = float(v)
            except:
                pass

        # 이동키 반영
        vx = 0.0
        vyaw = 0.0

        if self.key_states["w"]:
            vx += self.speed["linear"]
        if self.key_states["s"]:
            vx -= self.speed["linear"]
        if self.key_states["a"]:
            vyaw += self.speed["angular"]
        if self.key_states["d"]:
            vyaw -= self.speed["angular"]

        action["base.linear_velocity"] = vx
        action["base.angular_velocity"]     = vyaw
        return action

    # ================================================================
# PC 코드의 run 함수를 이걸로 덮어쓰세요
# ================================================================
    def run(self):
        print(f"\n🚀 RAW 기반 PC Sender Started! Dest: {args.rpi_ip}:{args.rpi_port}")
        
        # [핵심] 2초 동안 답 없으면 에러 발생시킴 (무한 멈춤 방지)
        self.sock.setsockopt(zmq.RCVTIMEO, 2000)
        self.sock.setsockopt(zmq.LINGER, 0)

        try:
            while not self.ctrl_c_pressed:
                if self.pending_event:
                    msg = self.pending_event
                    self.pending_event = None
                else:
                    msg = self.build_action()

                try:
                    # 1. 데이터 전송
                    self.sock.send_string(json.dumps(msg))
                    
                    # 2. 응답 대기 (여기서 멈추던 것임)
                    resp = self.sock.recv_string()
                    
                    # 3. 정상 수신 시 출력
                    if "event" in msg:
                        print(f"\n[EVENT SENT] {msg} -> Ack received")
                    else:
                        print(f"[ACTION] v={msg.get('base.linear_velocity', 0):.2f} RPi: Connected ✅", end="\r", flush=True)

                except zmq.Again:
                    # 2초간 응답 없으면 여기로 옴
                    print(f"\n⚠️ [Timeout] 라즈베리파이가 데이터를 받았지만 응답하지 않습니다.", end="\r")
                    
                    # 소켓 초기화 (재접속 시도)
                    self.sock.close()
                    self.sock = self.ctx.socket(zmq.REQ)
                    self.sock.connect(f"tcp://{args.rpi_ip}:{args.rpi_port}")
                    self.sock.setsockopt(zmq.RCVTIMEO, 2000)
                    continue
                    
                except zmq.ZMQError as e:
                    print(f"\n❌ ZMQ Error: {e}")
                    break
                
                time.sleep(0.03)

        finally:
            self.listener.stop()
            if self.leader: self.leader.disconnect()
            self.sock.close()
            self.ctx.term()


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rpi.ip", dest="rpi_ip", required=True)
    parser.add_argument("--rpi.port", dest="rpi_port", default=5555, type=int)
    parser.add_argument("--teleop.port", dest="teleop_port", default="/dev/ttyACM0")
    parser.add_argument("--teleop.id", dest="teleop_id", default=None)

    args = parser.parse_args()
    PCUnifiedSender(args).run()