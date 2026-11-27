"""非検証 Dev の学習区画だけで、裁定票の下限と滑らかな信頼度曲線を推定する。"""
import json
import math
import pickle
from collections import Counter
import numpy as np
from neurograph.metacog_v152 import ROOT, MODEL


def fit():
    """例 score=0.6 の発話を連続確率へ写す、正則化した一次ロジットを求める。"""
    model = pickle.loads((ROOT / "metacog_v151.pkl").read_bytes())
    rows = [r for r in json.loads((ROOT / "dev_answers_v151.json").read_bytes()) if r["group"] == "fit"]
    judges = json.loads((ROOT / "fit_judges_v152.json").read_bytes())
    # 意味票の下位四分位とスロット間の背景一致の中央値を下限にする。
    model["semantic_floor"] = float(np.quantile([j["j_sem"] for j in judges], .25))
    model["cross_floor"] = float(np.median([j["cross_slot"] for j in judges]))
    counts = Counter(r["mode"] for r in rows)
    w = np.array([len(rows)/(len(counts)*counts[r["mode"]]) for r in rows])
    x = np.array([r["score"] for r in rows])
    y = np.array([r["correct"] for r in rows])
    # 線形確率モデルの傾きを中央点でのロジットの傾きに変換する。
    # 一様な基準分布一組分の分散を加え、小標本での急峻化を抑える。
    center = np.average(x, weights=w)
    covariance = np.average((x-center)*y, weights=w)
    variance = np.average((x-center)**2, weights=w)
    slope = 4 * max(1e-6, (covariance + 1/12)/(variance + 1/12))
    prevalence = (np.average(y,weights=w) + .5)/2
    intercept = math.log(prevalence/(1-prevalence)) - slope*(center+.5)/2
    model["calibration"] = (slope, intercept)
    MODEL.write_bytes(pickle.dumps(model))
    stats = dict(fit=len(rows), modes=counts, slope=slope, intercept=intercept,
                 semantic_floor=model["semantic_floor"], cross_floor=model["cross_floor"],
                 tau_low=.45, tau_high=.75, source="dev_answers_v151.json / fit")
    (ROOT/"training_v152.json").write_text(json.dumps(stats,indent=2))


if __name__ == "__main__":
    fit()
