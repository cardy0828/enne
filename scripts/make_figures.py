"""results/ の保存JSONから README 用の図を描く。

使い方: python -X utf8 scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": ["Yu Gothic", "Meiryo", "Noto Sans CJK JP", "sans-serif"],
    "font.size": 11,
    "axes.edgecolor": "#c8c8c8",
    "axes.labelcolor": "#191919",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": "#666666",
    "ytick.color": "#666666",
    "axes.grid": True,
    "grid.color": "#ececec",
    "grid.linewidth": 0.8,
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "savefig.dpi": 200,
})
TEXT2 = "#666666"
S1, S2, S3 = "#257de0", "#f47230", "#00b894"
DOMAINS = [("easy_ja", "やさしい日本語", S1), ("aozora", "青空文庫", S2), ("wikipedia", "Wikipedia", S3)]


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def fig_growth():
    hist = load("results_v14_90m_6sys.json")["history"]
    x = [h["cum_chars_m"] for h in hist]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    nodes = [h["alive_hidden"] / 1e6 for h in hist]
    a.bar(x, nodes, width=9, color=S1, zorder=2)
    for xi, n in zip(x, nodes):
        a.text(xi, n + 0.05, f"{n:.2f}M", ha="center", va="bottom", fontsize=10, color="#191919")
    a.set_title("生存ノード数（百万）", loc="left", fontsize=12)
    a.set_xlabel("累計学習文字数（百万）")
    a.set_xticks(x)
    a.set_ylim(0, max(nodes) * 1.2)
    for key, label, c in DOMAINS:
        y = [h["domains"][key]["bpc_lossless"] for h in hist]
        b.plot(x, y, color=c, linewidth=2, marker="o", markersize=6, zorder=3)
        b.text(x[-1] + 2, y[-1], label, va="center", fontsize=10, color="#191919")
    b.set_title("評価BPC（小さいほど良い）", loc="left", fontsize=12)
    b.set_xlabel("累計学習文字数（百万）")
    b.set_xticks(x)
    b.set_xlim(x[0] - 5, x[-1] + 22)
    fig.suptitle("v1.4 継続学習：ノードが増減しながら成長する", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, -0.02, "出典: results/results_v14_90m_6sys.json（既存記憶を含む評価）", fontsize=9, color=TEXT2)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_growth.png", bbox_inches="tight")
    plt.close(fig)


def fig_cartridge():
    d = load("results_v20_engram.json")
    cc = d["custom_cartridge"]
    vals = [cc["before"]["bpc_lossless"], cc["mounted"]["bpc_lossless"], cc["unmounted"]["bpc_lossless"]]
    labels = ["装着前", "装着中", "取り外し後"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    bars = a.bar(labels, vals, color=[TEXT2, S1, TEXT2], width=0.6, zorder=2)
    for r, v in zip(bars, vals):
        a.text(r.get_x() + r.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=11, color="#191919")
    a.set_title("架空の設定文のBPC：記憶の着脱", loc="left", fontsize=12)
    a.set_ylim(0, 7)
    curve = d["wiki_scaling_curve"]
    x = [c["wiki_chars_m"] for c in curve]
    y = [c["wiki_val"]["bpc_lossless"] for c in curve]
    bare = d["stage0_bare"]["wikipedia"]["bpc_lossless"]
    b.axhline(bare, color=TEXT2, linewidth=1.2, linestyle="--", zorder=2)
    b.text(x[0], bare + 0.004, f"本体のみ {bare:.3f}", fontsize=10, color=TEXT2, va="bottom")
    b.plot(x, y, color=S3, linewidth=2, marker="o", markersize=6, zorder=3)
    for xi, yi in zip(x, y):
        b.text(xi, yi - 0.012, f"{yi:.3f}", ha="center", va="top", fontsize=9, color="#191919")
    b.set_title("Wikipedia記憶の収録量とBPC（本体は固定）", loc="left", fontsize=12)
    b.set_xlabel("記憶に収録した文字数（百万）")
    b.set_xticks(x)
    b.set_ylim(min(y) - 0.04, bare + 0.03)
    fig.suptitle("v2.0 外部記憶：本体を再学習せずに知識を足し引きする", x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.text(0.02, -0.02, "出典: results/results_v20_engram.json", fontsize=9, color=TEXT2)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_memory.png", bbox_inches="tight")
    plt.close(fig)


def fig_lead():
    s = load("results_v112_lead_qa.json")["ablation_summary"]
    conds = [("H0_Flat6gram", "6-gram検索"), ("H1_WikiLead", "冒頭文優先検索")]
    fig, a = plt.subplots(figsize=(7, 3.6))
    import numpy as np
    idx = np.arange(2)
    w = 0.36
    ev = [s[k]["ev_h"] for k, _ in conds]
    an = [s[k]["an_h"] for k, _ in conds]
    gt = s[conds[0][0]]["gt"]
    r1 = a.bar(idx - w / 2 - 0.01, ev, w, color=S1, label="検索した根拠に含まれる", zorder=2)
    r2 = a.bar(idx + w / 2 + 0.01, an, w, color=S2, label="回答に含まれる", zorder=2)
    for r, v in list(zip(r1, ev)) + list(zip(r2, an)):
        a.text(r.get_x() + r.get_width() / 2, v + 0.2, f"{v}/{gt}", ha="center", fontsize=11, color="#191919")
    a.set_xticks(idx, [l for _, l in conds])
    a.set_ylim(0, gt)
    a.set_ylabel("評価語句の一致数")
    a.legend(frameon=False, loc="upper left", fontsize=10)
    a.set_title("v11.2 記事冒頭文検索：小規模QA（6問・評価語句15個）", loc="left", fontsize=12, fontweight="bold")
    fig.text(0.02, -0.04, "出典: results/results_v112_lead_qa.json（語句の部分一致。正答率ではない）", fontsize=9, color=TEXT2)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_lead_qa.png", bbox_inches="tight")
    plt.close(fig)


def fig_modules():
    """部品ごとの学習・構築時間（各結果JSONの記録値）。"""
    v20 = load("results_v20_engram.json")
    v21 = load("results_v21_thought_engram.json")
    v60 = load("results_v60_role_learning.json")
    v80 = load("results_v80_sft.json")["train_stats"]
    v110 = load("results_v110_merge.json")
    v112 = load("results_v112_lead_qa.json")
    v121 = load("training_v121.json")
    v14 = load("results_v14_90m_6sys.json")["history"]
    body_sec = sum(h["sec"] for h in v14[1:]) / (len(v14) - 1)  # 1ラウンド=1,500万文字
    rows = [
        ("架空知識の記憶カートリッジ作成", v20["custom_cartridge"]["build_ms"] / 1000, "CPU"),
        ("対話（SFT）記憶の書き込み", v80["engram"]["sec"], ""),
        ("Wikipedia文章記憶の索引", v21["ep_stats"]["ep_wikipedia"]["build_sec"], "CPU"),
        ("2つのDAGの構造対応（統合）", v110["topology_stats"]["topology_sec"], ""),
        ("役割表現の学習", v60["train_stats"]["sec"], ""),
        ("記事冒頭文索引（17億文字を走査）", v112["lead_index_stats"]["sec"], "CPU"),
        ("記憶ゲートの学習", v121["train_sec"], "CPU・GPU不使用"),
        ("対話の読み出し適応（終了予測）", v80["readout"]["sec"], ""),
        ("本体の継続学習（1,500万文字）", body_sec, "GPU"),
    ]
    rows.sort(key=lambda r: r[1])
    fig, a = plt.subplots(figsize=(9, 4.6))
    ys = range(len(rows))
    for y, (label, sec, where) in zip(ys, rows):
        c = TEXT2 if "本体" in label else S1
        a.hlines(y, 1e-3, sec, color=c, linewidth=2, zorder=2)
        a.plot(sec, y, "o", color=c, markersize=8, zorder=3)
        txt = f"{sec*1000:.1f} ms" if sec < 1 else (f"{sec:.0f} 秒" if sec < 120 else f"約{sec/60:.0f}分")
        if where:
            txt += f"（{where}）"
        a.text(sec * 1.35, y, txt, va="center", fontsize=10, color="#191919")
    a.set_yticks(list(ys), [r[0] for r in rows])
    a.set_xscale("log")
    a.set_xlim(1e-3, 3e4)
    ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000]
    a.set_xticks(ticks, ["1ms", "10ms", "0.1秒", "1秒", "10秒", "100秒", "1000秒", "1万秒"])
    a.minorticks_off()
    a.set_xlabel("学習・構築にかかった時間（秒、対数軸）")
    a.grid(axis="y", visible=False)
    a.set_title("部品ごとに学習でき、足すときに全体を再学習しない", loc="left", fontsize=13, fontweight="bold")
    fig.text(0.02, -0.03, "出典: results/ の各JSONの記録値（版ごとに別の実験。RTX 3060 12GB 環境）", fontsize=9, color=TEXT2)
    fig.tight_layout()
    fig.savefig(OUT / "fig0_modules.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig_modules()
    fig_growth()
    fig_cartridge()
    fig_lead()
    print("saved to", OUT)
