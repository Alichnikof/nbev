"""README figures: nbev packing + attention mask (toy example), and the training curves of a run.

    uv run python scripts/readme_figures.py runs/<run-id> assets
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch

run_dir, out_dir = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- figure 1: packing + mask
context = ["[CLS]", "Which", "team", "?", "…", "Resp:", "…", "[SEP]"]
choices = [["[OPT]", "Billing", "[SEP]"], ["[OPT]", "Tech", "support", "[SEP]"], ["[OPT]", "Sales", "[SEP]"]]
n = len(context)
tokens, positions, owner = list(context), list(range(n)), [0] * n
for i, block in enumerate(choices, 1):
    tokens += block
    positions += range(n, n + len(block))
    owner += [i] * len(block)
L = len(tokens)

colors = ["#9aa5b1", "#3b82f6", "#f59e0b", "#10b981"]
labels = ["context (question + state)", "choice A", "choice B", "choice C"]

mask = [[0] * L for _ in range(L)]
for q in range(L):
    for k in range(L):
        if owner[k] == 0 or owner[q] == owner[k]:
            mask[q][k] = owner[q] + 1 if owner[q] == owner[k] else 5 if owner[q] else 1
# values: 0 blocked, 1 context<->context, 2..4 own block, 5 choice->context
cmap = ListedColormap(["#ffffff", colors[0], colors[1], colors[2], colors[3], "#d6dbe1"])

fig = plt.figure(figsize=(11, 9.2), dpi=150)
ax_seq = fig.add_axes([0.04, 0.80, 0.92, 0.17])
ax_seq.set_xlim(0, L)
ax_seq.set_ylim(0, 2.2)
ax_seq.axis("off")
for x, (tok, pos, o) in enumerate(zip(tokens, positions, owner)):
    ax_seq.add_patch(FancyBboxPatch((x + 0.06, 1.0), 0.88, 0.9, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=colors[o], ec="none", alpha=0.9))
    ax_seq.text(x + 0.5, 1.45, tok, ha="center", va="center", fontsize=7.5, color="white", weight="bold")
    ax_seq.text(x + 0.5, 0.65, str(pos), ha="center", va="center", fontsize=9,
                color=colors[o] if o else "#52606d", weight="bold" if o else "normal")
ax_seq.text(0, 0.15, "position id  →  every choice restarts at position %d" % n, fontsize=9, color="#52606d")
ax_seq.set_title("nbev: one packed sequence per question", fontsize=13, loc="left", weight="bold")

ax = fig.add_axes([0.12, 0.08, 0.58, 0.64])
ax.imshow(mask, cmap=cmap, vmin=0, vmax=5)
ax.set_xticks(range(L), tokens, rotation=90, fontsize=8)
ax.set_yticks(range(L), tokens, fontsize=8)
for t, o in zip(ax.get_xticklabels(), owner):
    t.set_color(colors[o] if o else "#3e4c59")
for t, o in zip(ax.get_yticklabels(), owner):
    t.set_color(colors[o] if o else "#3e4c59")
ax.set_xticks([x - 0.5 for x in range(L + 1)], minor=True)
ax.set_yticks([x - 0.5 for x in range(L + 1)], minor=True)
ax.grid(which="minor", color="#e4e7eb", linewidth=0.6)
ax.tick_params(which="minor", length=0)
ax.set_xlabel("key (attended to)", fontsize=9)
ax.set_ylabel("query (attending)", fontsize=9)
ax.set_title("attention mask (coloured = allowed, white = blocked)", fontsize=10)
for spine in ax.spines.values():
    spine.set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in colors] + [plt.Rectangle((0, 0), 1, 1, fc="#d6dbe1"),
                                                                   plt.Rectangle((0, 0), 1, 1, fc="white", ec="#9aa5b1")]
fig.legend(handles, labels + ["choice → context", "blocked: choices never see each other"],
           loc="center right", bbox_to_anchor=(0.99, 0.42), fontsize=8, frameon=False)
fig.savefig(f"{out_dir}/nbev_packing.png", facecolor="white")

# ---------------------------------------------------------------- figure 2: training curves
rows = [json.loads(line) for line in open(f"{run_dir}/metrics.jsonl")]
train = [r for r in rows if r["split"] == "train"]
val = [r for r in rows if r["split"] == "validation"]

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), dpi=150)
a1.plot([r["step"] for r in train], [r["loss"] for r in train], color="#9aa5b1", lw=1.2, label="train (10-step average)")
a1.plot([r["step"] for r in val], [r["loss"] for r in val], "o-", color="#3b82f6", lw=2, label="validation")
a1.set_title("loss (NLL)", fontsize=11)
a1.set_xlabel("optimizer step (16 questions each)")
a1.legend(frameon=False, fontsize=9)

steps = [r["step"] for r in val]
for key, name, c in [("accuracy_choice", "choice accuracy", "#3b82f6"), ("auc_noul", "yes/no AUC", "#f59e0b"),
                     ("accuracy_score", "score accuracy", "#10b981")]:
    a2.plot(steps, [r[key] for r in val], "o-", color=c, lw=2, label=name)
for y, c in [(0.33, "#3b82f6"), (0.5, "#f59e0b"), (0.255, "#10b981")]:
    a2.axhline(y, color=c, ls=":", lw=1)
a2.set_title("validation (dotted = chance / majority baseline)", fontsize=11)
a2.set_xlabel("optimizer step")
a2.set_ylim(0.2, 0.85)
a2.legend(frameon=False, fontsize=9, loc="upper left")
for a in (a1, a2):
    a.spines[["top", "right"]].set_visible(False)
    a.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{out_dir}/nbev_training.png", facecolor="white")
print("ok")
