"""コマンドラインから睡眠統合を1回実行する（v14.0）。"""
import json
from neurograph.gate_v140 import check_device
from neurograph.run_v112_lead_qa import load_brain


def main():
    """指定 GPU を照合し、判定と失敗項目を表示する。"""
    check_device()
    brain,_ = load_brain()
    try:
        result = brain.sleep_evolve()
        print(json.dumps(result,ensure_ascii=False))
    finally:
        brain.lead_hip.close()


if __name__=="__main__":
    main()
