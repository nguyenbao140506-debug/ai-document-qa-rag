"""Create original synthetic PDFs and independently authored evidence annotations.

All institutions, policies, amounts, and procedures below are fictional test data.
ReportLab invariant mode removes timestamp/random-ID differences across builds.
"""

import json
from hashlib import sha256

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph

from ragqa.config import ROOT

# (heading, question, evidence, expected answer, question, evidence, expected answer)
SPEC = [
    (
        "cedar-research.pdf",
        "Cedar Research Handbook",
        [
            (
                "Archive and access",
                "How long are Cedar research backups retained?",
                "Cedar research backups are retained for 45 days before automatic deletion.",
                "45 days",
                "Who approves access to the Cedar research archive?",
                "Access to the Cedar research archive requires approval from the data steward.",
                "data steward",
            ),
            (
                "Purchases and travel",
                "Above what amount do Cedar equipment purchases need two approvals?",
                "Cedar research equipment purchases above 600 credits require two approvals.",
                "600 credits",
                "When must Cedar travel receipts be submitted?",
                "Cedar travel receipts must be submitted within 12 working days of return.",
                "12 working days",
            ),
            (
                "Incidents and reviews",
                "How quickly must Cedar research incidents be reported?",
                "Cedar research incident reports must be filed within 4 hours of discovery.",
                "4 hours",
                "In which month is the Cedar annual dataset review held?",
                "The Cedar annual dataset review is held every September.",
                "September",
            ),
        ],
    ),
    (
        "harbor-observatory.pdf",
        "Harbor Observatory Operations",
        [
            (
                "Observations and storage",
                "How long does Harbor keep raw telescope observations?",
                "Harbor Observatory retains raw telescope observations for 180 days.",
                "180 days",
                "What is the Harbor telescope booking interval?",
                "Harbor telescope reservations are allocated in 90-minute blocks.",
                "90-minute",
            ),
            (
                "Safety and weather",
                "At what wind speed does the Harbor dome close?",
                "The Harbor observatory dome must close when wind speed exceeds 35 kilometers per hour.",
                "35 kilometers per hour",
                "Who gives clearance to reopen the Harbor dome?",
                "Only the Harbor duty astronomer can authorize reopening the dome.",
                "duty astronomer",
            ),
            (
                "Maintenance and visits",
                "How often is the Harbor primary mirror inspected?",
                "The Harbor primary mirror is inspected every 6 weeks.",
                "6 weeks",
                "What is the maximum size of a Harbor guided group?",
                "Harbor guided observatory visits are limited to 14 visitors per group.",
                "14 visitors",
            ),
        ],
    ),
    (
        "willow-garden.pdf",
        "Willow Botanical Garden Guide",
        [
            (
                "Seeds and storage",
                "What temperature is used for Willow seed storage?",
                "Willow seed collections are stored at minus 18 degrees Celsius.",
                "minus 18",
                "How often are Willow seed germination tests performed?",
                "Willow seed germination tests are performed every 24 months.",
                "24 months",
            ),
            (
                "Water and access",
                "When does Willow irrigate the greenhouse?",
                "Willow greenhouse irrigation starts at 05:30 each morning.",
                "05:30",
                "Who may enter the Willow quarantine room?",
                "Only trained plant health technicians may enter the Willow quarantine room.",
                "plant health technicians",
            ),
            (
                "Volunteers and samples",
                "How many training hours do Willow volunteers complete?",
                "Willow volunteers complete 8 hours of supervised training before independent work.",
                "8 hours",
                "How long are Willow soil samples retained?",
                "Willow soil samples are retained for 21 days after analysis.",
                "21 days",
            ),
        ],
    ),
    (
        "atlas-robotics.pdf",
        "Atlas Robotics Lab Manual",
        [
            (
                "Battery and charging",
                "At what battery level must Atlas robots return to charge?",
                "Atlas mobile robots return to the charging station when battery charge falls below 25 percent.",
                "25 percent",
                "Where may Atlas lithium batteries be charged?",
                "Atlas lithium batteries may be charged only inside the ventilated charging cabinet.",
                "ventilated charging cabinet",
            ),
            (
                "Motion and safety",
                "What is the Atlas robot speed limit near people?",
                "The Atlas robot speed limit within two meters of a person is 0.4 meters per second.",
                "0.4 meters per second",
                "Who resets the Atlas emergency stop after an incident?",
                "The Atlas laboratory safety officer must approve an emergency-stop reset after an incident.",
                "safety officer",
            ),
            (
                "Logs and calibration",
                "How long does Atlas retain robot telemetry?",
                "Atlas robot telemetry logs are retained for 60 days.",
                "60 days",
                "How frequently are Atlas force sensors calibrated?",
                "Atlas force sensors are calibrated after every 120 operating hours.",
                "120 operating hours",
            ),
        ],
    ),
    (
        "delta-coastal.pdf",
        "Delta Coastal Fieldwork Protocol",
        [
            (
                "Sampling and transport",
                "How far apart are Delta coastal sampling stations?",
                "Delta coastal sampling stations are spaced 250 meters apart along each transect.",
                "250 meters",
                "What temperature range is required for Delta water samples in transit?",
                "Delta water samples must remain between 2 and 6 degrees Celsius during transport.",
                "2 and 6 degrees Celsius",
            ),
            (
                "Weather and teams",
                "What minimum team size is required for Delta coastal surveys?",
                "Every Delta coastal survey team must include at least 3 trained members.",
                "3 trained members",
                "What happens to Delta surveys during a lightning warning?",
                "Delta coastal surveys are suspended immediately when a lightning warning is issued.",
                "suspended immediately",
            ),
            (
                "Reports and equipment",
                "When are Delta field reports due?",
                "Delta field reports are due within 48 hours after the survey ends.",
                "48 hours",
                "How frequently are Delta salinity probes checked against a reference?",
                "Delta salinity probes are checked against a reference solution every morning.",
                "every morning",
            ),
        ],
    ),
    (
        "maple-library.pdf",
        "Maple Library Service Policies",
        [
            (
                "Loans and renewals",
                "How long can Maple members borrow ordinary books?",
                "Maple Library members may borrow ordinary books for 28 days.",
                "28 days",
                "How many renewals does Maple allow for an unreserved book?",
                "Maple Library permits at most 2 renewals for a book without a waiting reservation.",
                "2 renewals",
            ),
            (
                "Rooms and archives",
                "What is the maximum Maple study room booking length?",
                "Maple study rooms can be reserved for a maximum of 3 hours per day.",
                "3 hours",
                "How much notice is required to consult Maple rare manuscripts?",
                "Researchers must request Maple rare manuscripts at least 5 working days before their visit.",
                "5 working days",
            ),
            (
                "Returns and accessibility",
                "What is the Maple grace period for overdue ordinary books?",
                "Maple Library provides a 2-day grace period for overdue ordinary books.",
                "2-day",
                "When should Maple accessible-format requests be submitted?",
                "Maple accessible-format requests should be submitted 10 working days before the material is needed.",
                "10 working days",
            ),
        ],
    ),
]

FILLER = [
    "Staff use this handbook to check routine responsibilities before starting work. "
    "The current page applies to the named institution only. Similar programs at other "
    "institutions may follow different schedules and procedures. A designated coordinator "
    "records completion in the activity register and asks the responsible specialist to "
    "resolve unclear entries. Team members verify the relevant section rather than relying "
    "on memory. This fictional example is intended for testing information retrieval and "
    "does not describe a real organization.",
    "Before an activity begins, the assigned team checks that its equipment and records are "
    "ready. Participants use the local checklist and report discrepancies to their supervisor. "
    "The register includes the activity date, the responsible person, and a concise description "
    "of any exception. A second member checks the entry when the procedure requires review. "
    "These administrative details support continuity between shifts and help new team members "
    "find the appropriate section of the manual.",
    "After completing the activity, staff review the record for missing information and file "
    "it under the relevant program. Questions about exceptions are referred to the coordinator "
    "rather than decided from an unrelated handbook. Temporary notes are reconciled with the "
    "official register during handover. Managers use the completed records to identify recurring "
    "operational issues. The handbook provides a shared reference for this fictional workflow "
    "and should be interpreted together with the heading on the current page.",
]


def main():
    corpus = ROOT / "evaluation" / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    questions, manifest = [], []
    style = ParagraphStyle(
        "body",
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#263552"),
    )
    for filename, title, pages in SPEC:
        path = corpus / filename
        pdf = canvas.Canvas(str(path), pagesize=(595, 842), invariant=1, pageCompression=1)
        pdf.setTitle(title)
        pdf.setAuthor("AI Document QA synthetic evaluation corpus")
        for number, (heading, q1, e1, a1, q2, e2, a2) in enumerate(pages, 1):
            pdf.setFillColor(colors.HexColor("#18294b"))
            pdf.rect(0, 718, 595, 124, fill=1, stroke=0)
            pdf.setFillColor(colors.white)
            pdf.setFont("Helvetica-Bold", 19)
            for line, text in enumerate(simpleSplit(title, "Helvetica-Bold", 19, 495)):
                pdf.drawString(50, 790 - line * 24, text)
            pdf.setFont("Helvetica", 12)
            pdf.drawString(50, 742, heading)
            y = 685
            blocks = [FILLER[0] + " " + e1, FILLER[1], e2 + " " + FILLER[2]]
            for text in blocks:
                paragraph = Paragraph(text, style)
                _, height = paragraph.wrap(495, 600)
                if y - height < 75:
                    raise RuntimeError(f"Synthetic page content overflowed: {filename}:{number}")
                paragraph.drawOn(pdf, 50, y - height)
                y -= height + 18
            pdf.setFillColor(colors.HexColor("#64748b"))
            pdf.setFont("Helvetica", 9)
            pdf.drawString(50, 42, "SYNTHETIC EVALUATION DATA - Original fictional content")
            pdf.drawRightString(545, 42, f"Page {number}")
            pdf.showPage()
            for question, evidence, answer in [(q1, e1, a1), (q2, e2, a2)]:
                questions.append(
                    {
                        "id": f"q{len(questions) + 1:02d}",
                        "question": question,
                        "filename": filename,
                        "page": number,
                        "evidence": evidence,
                        "expected_answer": answer,
                        "answerable": True,
                        "split": "evaluation",
                    }
                )
        pdf.save()
        manifest.append(
            {
                "filename": filename,
                "sha256": sha256(path.read_bytes()).hexdigest(),
                "pages": len(pages),
            }
        )
    negatives = [
        "What is the capital city of France?",
        "What is the administrator password for the Cedar archive?",
        "How much is the annual membership fee at Maple Library?",
        "Who founded the Harbor Observatory?",
        "What is the exact GPS location of the Willow garden?",
        "What brand of processor powers the Atlas robots?",
        "What is the Delta project's annual budget?",
        "Which country won the 2022 football World Cup?",
    ]
    questions += [
        {"id": f"n{i:02d}", "question": q, "answerable": False, "split": "evaluation"}
        for i, q in enumerate(negatives, 1)
    ]
    (ROOT / "evaluation" / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (ROOT / "evaluation" / "corpus_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Created {len(SPEC)} original PDFs, 18 pages, 36 answerable and 8 unanswerable questions.")


if __name__ == "__main__":
    main()
