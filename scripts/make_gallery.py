"""figures/index.html — every image under figures/, grouped by folder (open it in a browser)."""
import html
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
TITLES = {
    "data": "Data views", "data/geophysics": "Geophysics maps with labels", "detections": "Model detections (OOF)",
    "org_class_descriptions": "Organisers: class descriptions", "org_consultation1": "Organisers: consultation 1",
    "org_consultation2": "Organisers: consultation 2", "org_archaeologists_qa": "Organisers: archaeologists' Q&A",
    "solution_strategy": "Solution strategy (team PDF)",
}


def main():
    groups = {}
    for dp, _, fs in os.walk(ROOT):
        imgs = sorted(f for f in fs if f.lower().endswith((".png", ".jpg", ".jpeg", ".svg")))
        if imgs:
            groups[os.path.relpath(dp, ROOT)] = imgs
    order = sorted(groups, key=lambda g: (g.startswith("org") or g.startswith("solution"), g))
    parts = ["""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>KOZ-3 Figures</title><style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--bd:#e3e2dc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9993;--card:#21211f;--bd:#33332f}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}
main{max-width:1400px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
nav a{color:var(--mut);margin-right:14px;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--bd);border-radius:8px;overflow:hidden}
figure img{width:100%;display:block;cursor:zoom-in}figcaption{font-size:12px;color:var(--mut);padding:6px 8px;word-break:break-all}
</style></head><body><main><h1>KOZ-3 figures</h1><nav>"""]
    parts += [f'<a href="#{html.escape(g)}">{html.escape(TITLES.get(g, g))} ({len(groups[g])})</a>' for g in order]
    parts.append("</nav>")
    for g in order:
        parts.append(f'<h2 id="{html.escape(g)}">{html.escape(TITLES.get(g, g))}</h2><div class="grid">')
        for f in groups[g]:
            src = html.escape(f"{g}/{f}" if g != "." else f)
            parts.append(f'<figure><a href="{src}" target="_blank"><img loading="lazy" src="{src}"></a><figcaption>{html.escape(f)}</figcaption></figure>')
        parts.append("</div>")
    parts.append("</main></body></html>")
    open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write("\n".join(parts))
    print(sum(len(v) for v in groups.values()), "images in", len(groups), "groups")


if __name__ == "__main__":
    main()
