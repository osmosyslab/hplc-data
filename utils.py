import webbrowser
from pathlib import Path

def visualizar(df, titulo="Resultado", nombre="output"):
    root = Path(__file__).parent

    for old in root.glob(f"{nombre}*.html"):
        old.unlink()

    html = df.to_html(index=False, border=1, classes="table")
    page = f"""
    <html><head>
    <meta charset="utf-8">
    <title>{titulo}</title>
    <style>
        body {{ font-family: sans-serif; padding: 20px; }}
        .table {{ border-collapse: collapse; width: 100%; }}
        .table th, .table td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: right; }}
        .table th {{ background: #f0f0f0; text-align: center; }}
    </style>
    </head><body>
    <h2>{titulo}</h2>
    {html}
    </body></html>
    """
    out = root / f"{nombre}.html"
    out.write_text(page, encoding="utf-8")
    webbrowser.open_new_tab(out.as_uri())
