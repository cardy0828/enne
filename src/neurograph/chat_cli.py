"""保存済みの統合モデルを使う対話端末。"""
import argparse
import contextlib
import json
import math
import sys
import time

import torch

from neurograph.instruct_cortex_v80 import _mem_snapshot
from neurograph.run_v112_lead_qa import load_brain


def main():
    """例として --once と --trace を併用すると、一問の回答と検索根拠を表示する。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once")
    parser.add_argument("--mount-text")
    parser.add_argument("--no-hip", action="store_true")
    parser.add_argument("--single-gate", action="store_true")
    parser.add_argument("--arbiter-mode",choices=("arbiter","consensus","single"),default="arbiter")
    parser.add_argument("--no-utterance-nodes",action="store_true")
    parser.add_argument("--tool-axis-mode",choices=("orthogonal","mixed","off"),default="orthogonal")
    parser.add_argument("--no-tools",action="store_true")
    parser.add_argument("--no-metacog", action="store_true")
    parser.add_argument("--confidence", action="store_true")
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--temp", type=float, default=0.)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    if not math.isfinite(args.temp) or args.temp < 0 or not 0 < args.top_p <= 1:
        parser.error("温度は有限の非負値、top-p は (0, 1] で指定してください")
    from neurograph.gate_v140 import check_device
    check_device()
    started = time.perf_counter()
    with contextlib.redirect_stdout(sys.stderr):
        brain, _ = load_brain()
    startup_sec = round(time.perf_counter() - started, 2)
    mounted_mib = round(torch.cuda.memory_allocated(brain.device) / 2**20)
    history = []
    enabled = not args.no_hip
    gate_enabled = True
    trace = args.trace
    if args.mount_text:
        brain.lead_hip.mount_custom("cli", args.mount_text)
    try:
        while True:
            try:
                line = args.once if args.once is not None else input("> ").strip()
            except EOFError:
                break
            if args.once is None and line.startswith("/"):
                command, *parts = line.split(maxsplit=2)
                if command == "/quit":
                    break
                if command in ("/teach","/correct","/sleep","/generations","/rollback"):
                    try:
                        if command=="/teach" and len(parts)==2:
                            brain.teach_fact(*parts)
                        elif command=="/correct" and parts and history:
                            brain.record_correction(history[-1][0],parts[0],parts[1] if len(parts)>1 else "")
                        elif command=="/sleep" and not parts:
                            result = brain.sleep_evolve()
                            print(result["status"],result["reasons"])
                        elif command=="/generations" and not parts:
                            print(brain.generation_id,sorted(p.name for p in brain.evolution.gens.glob("gen_*")))
                        elif command=="/rollback" and len(parts)==1:
                            brain.evolution.rollback(brain,parts[0])
                            history.clear()
                        else:
                            raise ValueError("/teach 主題 本文、直前の発話に /correct 種別 [対象]、/sleep、/generations、/rollback 世代")
                    except (ValueError,FileNotFoundError) as error:
                        print(error)
                    continue
                if command == "/reset":
                    history.clear()
                elif command == "/temp" and len(parts) == 1:
                    try:
                        value = float(parts[0])
                        if not math.isfinite(value) or value < 0:
                            raise ValueError
                        args.temp = value
                        print("temperature", value)
                    except ValueError:
                        print("有限の非負値を指定してください")
                elif command == "/metacog" and parts in (["on"], ["off"]):
                    args.no_metacog = parts == ["off"]
                elif command == "/confidence" and parts in (["on"], ["off"]):
                    args.confidence = parts == ["on"]
                elif command == "/trace":
                    trace = not trace
                    print("trace", trace)
                elif command == "/utterance" and parts in (["on"], ["off"]):
                    args.no_utterance_nodes = parts == ["off"]
                elif command == "/tools" and parts in (["on"], ["off"]):
                    args.no_tools = parts == ["off"]
                elif command == "/gate" and parts in (["on"], ["off"]):
                    gate_enabled = parts == ["on"]
                elif command == "/hip" and parts in (["on"], ["off"]):
                    enabled = parts == ["on"]
                elif command == "/mount" and len(parts) == 2:
                    brain.lead_hip.mount_custom(*parts)
                elif command == "/unmount" and len(parts) == 1:
                    brain.lead_hip.unmount_custom(parts[0])
                else:
                    print("/reset /temp <float> /trace /utterance on|off /tools on|off /gate on|off /hip on|off /mount <name> <text> /unmount <name> /quit")
                continue
            if line.strip():
                answer, valid, info = brain.respond(line, use_metacog=not args.no_metacog, record_episode=True, history=history, hip_enabled=enabled, score_critic=False, use_memory_gate=gate_enabled,
                                                    use_utterance_nodes=not args.no_utterance_nodes,use_tools=not args.no_tools, tool_axis_mode=args.tool_axis_mode, arbiter_mode="single" if args.single_gate else args.arbiter_mode, temperature=args.temp, top_p=args.top_p, seed=args.seed)
                print(answer)
                if args.confidence:
                    audit = info["metacog_trace"]
                    print(json.dumps({k:audit[k] for k in ("confidence_band","final_confidence","revisions")}))
                if trace:
                    print(json.dumps(dict(info, startup_sec=startup_sec, mounted_cuda_mib=mounted_mib, valid_utf8=valid, memory=_mem_snapshot(brain.device),
                                          peak_cuda_alloc_mb=torch.cuda.max_memory_allocated() / 2**20), ensure_ascii=False))
                history = (history + [(line, answer)])[-2:]
            if args.once is not None:
                break
    finally:
        brain.lead_hip.close()


if __name__ == "__main__":
    main()
