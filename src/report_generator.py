from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime

def generate_report(prediction, confidence, probabilities):

    doc = SimpleDocTemplate("Prediction_Report.pdf")

    styles = getSampleStyleSheet()

    story = []

    story.append(Paragraph("<b>LungVision AI</b>", styles["Title"]))

    story.append(Paragraph("Chest X-ray Disease Detection Report", styles["Heading2"]))

    story.append(Spacer(1,20))

    story.append(Paragraph(
        f"<b>Date:</b> {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}",
        styles["Normal"]
    ))

    story.append(Spacer(1,12))

    story.append(Paragraph(
        f"<b>Prediction:</b> {prediction}",
        styles["Normal"]
    ))

    story.append(Paragraph(
        f"<b>Confidence:</b> {confidence:.2f}%",
        styles["Normal"]
    ))

    story.append(Spacer(1,20))

    story.append(Paragraph("<b>Class Probabilities</b>", styles["Heading2"]))

    for cls, value in probabilities.items():
        story.append(
            Paragraph(
                f"{cls}: {value:.2f}%",
                styles["Normal"]
            )
        )

    story.append(Spacer(1,20))

    story.append(
        Paragraph(
            "Generated using LungVision AI",
            styles["Italic"]
        )
    )

    doc.build(story)