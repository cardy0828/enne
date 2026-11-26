"""ENNE の考え方を示す簡略サンプル（研究本体の実装ではありません）。

依存ライブラリなし、CPU だけで数秒で動きます。

    python examples/toy_enne.py

示すこと:
  1. 256 バイトのノードだけから始まり、予測を外した所にノードが生まれ、
     使われないノードが削除される（誤差逆伝播なし）
  2. 知識を本体の外の「記憶カートリッジ」に置き、再学習なしで着脱できる
  3. 別々に育てたグラフでも、ノードが経路で特定できるので 1 つに統合できる

研究本体は、連続表現・作業記憶・GPU カーネル・ゲート学習などを含む別物です。
ここでは仕組みの骨格だけを、可変長の文脈統計として素朴に実装しています。
"""
from __future__ import annotations

import math
from collections import Counter

MAX_DEPTH = 8          # 文脈としてさかのぼる最大バイト数
BIRTH_SURPRISE = 24.0  # 葉ノードに溜まった驚き（bits）がこれを超えたら子ノードを作る
MIN_USES = 3           # 剪定時、これ未満しか使われていない葉ノードを削除する


class Node:
    __slots__ = ("counts", "total", "surprise", "uses", "children")

    def __init__(self) -> None:
        self.counts: Counter[int] = Counter()
        self.total = 0
        self.surprise = 0.0
        self.uses = 0
        self.children = 0


class GrowingByteGraph:
    """ノードは「直前のバイト列を新しい順に並べた経路」で特定される。"""

    def __init__(self) -> None:
        # 感覚ノード: 256 バイトそれぞれに 1 つ。中間ノードは 0 個から始まる
        self.nodes: dict[bytes, Node] = {bytes([b]): Node() for b in range(256)}
        self.births = 0
        self.deaths = 0

    def chain(self, ctx: bytes) -> list[tuple[bytes, Node]]:
        """文脈に一致するノードを浅い順にたどる。"""
        out = []
        for k in range(1, min(MAX_DEPTH, len(ctx)) + 1):
            path = ctx[-k:][::-1]
            node = self.nodes.get(path)
            if node is None:
                break
            out.append((path, node))
        return out

    @staticmethod
    def mix(chain: list[tuple[bytes, Node]], y: int) -> float:
        """浅いノードから深いノードへ、観測数に応じて予測を上書きしていく。"""
        p = 1.0 / 256
        for _, node in chain:
            if node.total == 0:
                continue
            lam = node.total / (node.total + 2.0 * len(node.counts) + 1.0)
            p = lam * (node.counts[y] / node.total) + (1.0 - lam) * p
        return p

    def prob(self, ctx: bytes, y: int) -> float:
        return self.mix(self.chain(ctx), y)

    def learn(self, ctx: bytes, y: int) -> float:
        ch = self.chain(ctx)
        p = self.mix(ch, y)
        # 局所更新: たどったノードの統計だけを書き換える
        for _, node in ch:
            node.counts[y] += 1
            node.total += 1
            node.uses += 1
        # 新生: 葉ノードで驚きが溜まったら、1 バイト深い子ノードを作る
        leaf_path, leaf = ch[-1]
        leaf.surprise += -math.log2(p)
        depth = len(leaf_path)
        if leaf.surprise > BIRTH_SURPRISE and depth < MAX_DEPTH and len(ctx) > depth:
            child = leaf_path + ctx[-(depth + 1)].to_bytes(1, "big")
            if child not in self.nodes:
                self.nodes[child] = Node()
                leaf.children += 1
                self.births += 1
            leaf.surprise = 0.0
        return p

    def prune(self) -> None:
        """削除: あまり使われていない葉ノードを取り除く（感覚ノードは残す）。"""
        dead = [k for k, n in self.nodes.items() if len(k) > 1 and n.children == 0 and n.uses < MIN_USES]
        for k in dead:
            del self.nodes[k]
            self.nodes[k[:-1]].children -= 1
        self.deaths += len(dead)
        for n in self.nodes.values():
            n.uses = 0

    def hidden(self) -> int:
        return len(self.nodes) - 256


class MemoryCartridge:
    """本体の外に置く記憶。文章から文脈→次バイトの統計を作るだけ。"""

    def __init__(self, text: str, orders: tuple[int, ...] = (6, 4, 3)) -> None:
        data = text.encode("utf-8")
        self.orders = orders
        self.table: dict[bytes, Counter[int]] = {}
        for i in range(len(data)):
            for o in orders:
                if i >= o:
                    self.table.setdefault(data[i - o : i], Counter())[data[i]] += 1

    def lookup(self, ctx: bytes) -> Counter[int] | None:
        for o in self.orders:  # 長い文脈から順に探す
            hit = self.table.get(ctx[-o:]) if len(ctx) >= o else None
            if hit:
                return hit
        return None


class Model:
    def __init__(self, core: GrowingByteGraph) -> None:
        self.core = core
        self.mounted: list[MemoryCartridge] = []

    def prob(self, ctx: bytes, y: int) -> float:
        p = self.core.prob(ctx, y)
        for cart in self.mounted:
            hit = cart.lookup(ctx)
            if hit:
                total = sum(hit.values())
                gate = max(hit.values()) / total * total / (total + 1.0)  # 確信度に応じたゲート
                p = (1.0 - gate) * p + gate * (hit[y] / total)
        return max(p, 1e-9)


def train(g: GrowingByteGraph, text: str) -> None:
    data = text.encode("utf-8")
    for i in range(1, len(data)):
        g.learn(data[max(0, i - MAX_DEPTH) : i], data[i])


def bpc(model: Model, text: str) -> float:
    """1 文字あたりのビット数（小さいほど良い）。"""
    data = text.encode("utf-8")
    bits = sum(-math.log2(model.prob(data[max(0, i - MAX_DEPTH) : i], data[i])) for i in range(1, len(data)))
    return bits / max(1, len(text) - 1)


def merge(a: GrowingByteGraph, b: GrowingByteGraph) -> tuple[GrowingByteGraph, float]:
    """同じ経路のノードは統計を足し合わせ、片方にしかないノードは移植する。"""
    m = GrowingByteGraph()
    m.nodes = {}
    for src in (a, b):
        for path, node in src.nodes.items():
            dst = m.nodes.setdefault(path, Node())
            dst.counts.update(node.counts)
            dst.total += node.total
    for path in m.nodes:
        if len(path) > 1 and path[:-1] in m.nodes:
            m.nodes[path[:-1]].children += 1
    shared = sum(1 for k in b.nodes if len(k) > 1 and k in a.nodes)
    return m, shared / max(1, b.hidden())


# ---- サンプル用の短い文章（このサンプルのために書いたもの） ----
PROSE = """
朝の駅には、仕事へ向かう人と学校へ向かう人が集まる。電車が来ると、人々は静かに乗り込み、窓の外の景色を眺める。
川沿いの道には桜の木が並び、春になると多くの人が散歩に訪れる。夏には子どもたちが川で遊び、秋には木の葉が赤く色づく。
図書館では、本を読む人や勉強をする人が静かに過ごしている。新しい本が入ると、入口の棚に並べられる。
町の市場では、野菜や果物、魚などが売られている。店の人は元気な声で客を呼び、客は品物を選んで買っていく。
""" * 6
PROSE_TEST = "川沿いの図書館には、春になると新しい本が並べられる。駅の市場では、朝から野菜や魚が売られている。"

DIALOGUE = """
ユーザー: こんにちは。今日はいい天気ですね。
アシスタント: こんにちは。本当にいい天気ですね。散歩に出かけるのもいいかもしれません。
ユーザー: おすすめの過ごし方はありますか。
アシスタント: 川沿いを歩いたり、図書館で本を読んだりするのはいかがでしょうか。
ユーザー: ありがとうございます。
アシスタント: どういたしまして。楽しい一日を過ごしてください。
""" * 6
DIALOGUE_TEST = "ユーザー: こんにちは。おすすめの本はありますか。\nアシスタント: こんにちは。図書館で新しい本を探すのはいかがでしょうか。"

FICTION = "量子重力エンジン「アストラX-99」は、2042年につくば研究所で開発された推進装置である。最大推力は従来のイオンエンジンの約400倍で、火星までの航行を18日間に短縮できる。"


def main() -> None:
    print("1. ゼロから育てる（誤差逆伝播なし）")
    g = GrowingByteGraph()
    print(f"   開始時: 感覚ノード 256 / 中間ノード {g.hidden()}")
    for r in range(1, 5):
        train(g, PROSE)
        g.prune()
        print(f"   ラウンド{r}: 中間ノード {g.hidden():5d}（累計 新生 {g.births:5d} / 削除 {g.deaths:5d}）"
              f"  未見文のBPC {bpc(Model(g), PROSE_TEST):.2f}")

    print("\n2. 知識を本体の外に置き、着脱する（本体は再学習しない）")
    model = Model(g)
    before = bpc(model, FICTION)
    model.mounted.append(MemoryCartridge(FICTION))
    mounted = bpc(model, FICTION)
    model.mounted.clear()
    after = bpc(model, FICTION)
    print(f"   架空の設定文のBPC  装着前 {before:.2f} → 装着中 {mounted:.2f} → 取り外し後 {after:.2f}")

    print("\n3. 別々に育てたグラフを 1 つに統合する")
    prose_g, dlg_g = GrowingByteGraph(), GrowingByteGraph()
    for _ in range(3):
        train(prose_g, PROSE); prose_g.prune()
        train(dlg_g, DIALOGUE); dlg_g.prune()
    merged, overlap = merge(prose_g, dlg_g)
    print(f"   対話グラフの中間ノードのうち、文章グラフと同じ経路を持つもの: {overlap:.1%}")
    print(f"   {'':14s}{'文章のBPC':>10s}{'対話のBPC':>10s}")
    for name, gg in (("文章だけ", prose_g), ("対話だけ", dlg_g), ("統合後", merged)):
        print(f"   {name:14s}{bpc(Model(gg), PROSE_TEST):10.2f}{bpc(Model(gg), DIALOGUE_TEST):10.2f}")


if __name__ == "__main__":
    main()
