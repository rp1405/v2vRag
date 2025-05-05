from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_JUSTIFY
import os


def convert_text_to_pdf(text_content, output_path="output.pdf"):
    """
    Convert text content to PDF with proper formatting.

    Args:
        text_content (str): The text content to convert
        output_path (str): Path where the PDF will be saved
    """
    # Create a PDF document
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )

    # Create styles
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(name="Justify", alignment=TA_JUSTIFY, fontSize=12, leading=14)
    )

    # Prepare the content
    story = []

    # Split the text into paragraphs and add them to the story
    paragraphs = text_content.split("\n\n")
    for para in paragraphs:
        if para.strip():  # Only add non-empty paragraphs
            p = Paragraph(para.strip(), styles["Justify"])
            story.append(p)
            story.append(Spacer(1, 12))  # Add space between paragraphs

    # Build the PDF
    doc.build(story)

    return output_path


if __name__ == "__main__":
    # Example usage
    sample_text = """This is a sample text.
    
    This is another paragraph.
    
    And one more paragraph."""

    output_file = convert_text_to_pdf(sample_text)
    print(f"PDF created successfully at: {os.path.abspath(output_file)}")
