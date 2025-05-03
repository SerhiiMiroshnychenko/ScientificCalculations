from pdf2docx import Converter

pdf_file = 'Практичні-тема-2.pdf'
docx_file = 'Практичні-тема-2.docx'
cv = Converter(pdf_file)
cv.convert(docx_file)
cv.close()
