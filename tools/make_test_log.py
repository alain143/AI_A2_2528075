"""Run pytest for real and save evidence/pytest_log.txt and evidence/AI_A2_<REGDIGITS>_TEST_LOG.pdf.

    pip install reportlab      # documentation tool only
    python tools/make_test_log.py
"""
import subprocess, sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
res = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=ROOT, capture_output=True, text=True)
text = res.stdout + res.stderr
(ROOT / "evidence" / "pytest_log.txt").write_text(text, encoding="utf-8")
pdf = ROOT / "evidence" / "AI_A2_2528075_TEST_LOG.pdf"
c = canvas.Canvas(str(pdf), pagesize=A4); y = 800
c.setFont("Courier", 7.5)
for line in ["AI-A2-8075 | IGIRANEZA Alain Providence | 25/28075 | pytest -v", ""] + text.splitlines():
    if y < 40: c.showPage(); c.setFont("Courier", 7.5); y = 800
    c.drawString(30, y, line[:130]); y -= 9.5
c.save(); print(text[-400:]); print("wrote", pdf); sys.exit(res.returncode)
