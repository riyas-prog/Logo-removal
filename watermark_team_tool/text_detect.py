from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en"
)

result = ocr.predict("tst.png")

print("\nDetected Text:\n")

for page in result:
    for text in page["rec_texts"]:
        print(text)