"""Create evidence/AI_A2_<REGDIGITS>_DECLARATION.pdf (print, sign, scan/photo if required)."""
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
pdf = ROOT / "evidence" / "AI_A2_2528075_DECLARATION.pdf"
c = canvas.Canvas(str(pdf), pagesize=A4)
c.setFont("Helvetica-Bold", 14); c.drawString(50, 780, "Student Declaration - SWE 3513 Individual Assignment 2")
c.setFont("Helvetica", 11)
lines = ["Name: IGIRANEZA Alain Providence", "Registration number: 25/28075", "Verification code: AI-A2-8075", "",
         "I declare that I am the author of this submission and that I understand it.",
         "I have disclosed all AI assistance in evidence/AI_USE.md.",
         "I can explain every file and make a live modification when asked.",
         "My dataset, seed, metrics and plots follow my own registration digits.",
         "I have not copied another student's fingerprint, code or explanations.", "",
         "Signature: ______________________________   Date: ________________"]
y = 740
for l in lines: c.drawString(50, y, l); y -= 24
c.save(); print("wrote", pdf)
