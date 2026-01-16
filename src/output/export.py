from docx import Document
from fpdf import FPDF

def save_docx(text, path):
    doc = Document()
    doc.add_paragraph(text)
    doc.save(path)

def save_pdf(text, path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(path)
