from pdf2docx import Converter

pdf_file = 'Практична робота 2 Теплотехніка.pdf'
docx_file = 'Практична робота 2.docx'
cv = Converter(pdf_file)
cv.convert(docx_file)
cv.close()
