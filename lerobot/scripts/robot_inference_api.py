from fastapi import FastAPI
from contextlib import asynccontextmanager
import threading
import argparse

# pc_inference_server 파일이 같은 경로에 있어야 합니다.
from pc_inference_server import InferenceServerController

controller: InferenceServerController | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- 시작(Startup) 로직 ---
    global controller

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy.path", dest="policy_path", required=True)
    parser.add_argument("--port", default=5555, type=int)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--task", default="Idle")
    
    # uvicorn 실행 인자와 섞여있을 수 있으므로 parse_known_args 사용
    args, _ = parser.parse_known_args()

    controller = InferenceServerController(args)

    # ZMQ + VLA 메인 루프를 백그라운드 스레드에서 실행
    t = threading.Thread(target=controller.run, daemon=True)
    t.start()
    print("✅ Inference loop started in background thread.")
    
    yield  # 서버 실행 중에는 여기서 대기
    
    # --- 종료(Shutdown) 로직 ---
    print("🛑 Server shutting down...")

# lifespan 파라미터 적용
app = FastAPI(lifespan=lifespan)

@app.post("/run_task")
async def run_task(body: dict):
    global controller
    if controller is None:
        return {"status": "error", "message": "controller not ready"}

    task = body.get("task")
    if not task:
        return {"status": "error", "message": "task is required"}

    controller.set_task(task)
    # (추후 필요하면 여기서 episode reset 신호도 추가 가능)
    return {"status": "ok", "task": task}

@app.get("/health")
async def health():
    return {"status": "ok"}

# 실행 커맨드 예시:
# python 이파일이름.py --host 0.0.0.0 --port 9000 --policy.path /경로/모델 --use_gpu
if __name__ == "__main__":
    import uvicorn
    # argparse로 받은 port를 사용하고 싶다면 args 파싱을 main에서 한 번 더 하거나
    # 고정 포트를 사용해야 합니다. 여기서는 편의상 9000으로 예시를 듭니다.
    # 실제 실행 시에는 터미널에서 python 파일명.py ... 로 실행하면 아래 로직이 돕니다.
    
    # 주의: 위 lifespan 안에서 argparse를 또 하기 때문에, 
    # main 실행 시 인자를 넘겨주면 lifespan 내부에서도 잘 받아옵니다.
    uvicorn.run(app, host="0.0.0.0", port=8080)