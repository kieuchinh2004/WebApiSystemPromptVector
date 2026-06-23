"""
main.py
Điểm khởi chạy ứng dụng WebAPI_Vector.
Chạy: python main.py
"""
import sys
import uvicorn
import config as cfg

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    print("=" * 65)
    print("   APC v4 VECTOR-BASED PROMPT CLASSIFIER API v2.0 (LLM + Math Engine)")
    print("=" * 65)
    print(f"  API       : http://{cfg.API_HOST}:{cfg.API_PORT}")
    print(f"  Docs      : http://127.0.0.1:{cfg.API_PORT}/docs")
    print(f"  Backend   : http://127.0.0.1:{cfg.LLAMA_PORT} (llama-server)")
    print(f"  Model     : {cfg.MODEL_NAME}")
    print(f"  Profile   : {cfg.MODEL_CONFIG_FILE}")
    print(f"  Prompt    : {cfg.PROMPT_FILE}")
    print("=" * 65)

    uvicorn.run(
        "app:app",
        host=cfg.API_HOST,
        port=cfg.API_PORT,
        reload=False,
        log_level="info",
    )
