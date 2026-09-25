"""
Generate a professional PDF summary explaining the Business Entity Resolution Project,
its architecture, and how each technology is used in simple language.
"""
import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)

def build_pdf(filename="Project_Summary_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        leading=26,
        textColor=colors.HexColor('#1A365D'),
        alignment=1, # Center
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#4A5568'),
        alignment=1,
        spaceAfter=15
    )
    
    h1_style = ParagraphStyle(
        'H1',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#2B6CB0'),
        spaceBefore=12,
        spaceAfter=6
    )
    
    h2_style = ParagraphStyle(
        'H2',
        parent=styles['Heading3'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=15,
        textColor=colors.HexColor('#2D3748'),
        spaceBefore=8,
        spaceAfter=4
    )
    
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=6
    )
    
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )
    
    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#2D3748')
    )
    
    table_cell_bold = ParagraphStyle(
        'TableCellBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold',
        textColor=colors.HexColor('#1A202C')
    )
    
    elements = []
    
    # Header
    elements.append(Paragraph("ML Challenge 2026: Business Entity Resolution", title_style))
    elements.append(Paragraph("<b>End-to-End System Architecture & Technology Breakdown (Simplified Guide)</b>", subtitle_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#2B6CB0'), spaceAfter=12))
    
    # 1. Project Overview
    elements.append(Paragraph("1. What is this Project About?", h1_style))
    elements.append(Paragraph(
        "<b>The Problem:</b> In commercial platforms, information about companies comes from multiple independent data sources (Source 1, Source 2, and Source 3). "
        "Each source has messy names (typos, abbreviations like 'Pvt Ltd' vs 'Limited'), partial addresses, and different formats, with <b>no common ID numbers</b>.",
        body_style
    ))
    elements.append(Paragraph(
        "<b>Our Goal:</b> For each reference business in Source 1 (over 1.73 Million entities), find all matching records in Source 2 and Source 3 (over 9.97 Million candidates) with high precision.",
        body_style
    ))
    elements.append(Paragraph(
        "<b>The Scale Challenge:</b> Comparing 1.73M records against 10M records directly would require <b>17 Trillion comparisons</b>, which would take weeks and crash any computer. "
        "Our solution solves this in minutes using an intelligent 4-stage pipeline.",
        body_style
    ))
    
    # 2. Pipeline Stages
    elements.append(Paragraph("2. How the Solution Works (4 Stages)", h1_style))
    
    pipeline_data = [
        [
            Paragraph("<b>Stage</b>", table_cell_bold),
            Paragraph("<b>What it Does</b>", table_cell_bold),
            Paragraph("<b>Why it's Crucial</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>1. Fast Candidate Blocking</b>", table_cell_bold),
            Paragraph("Builds an inverted token index and country filter to find the Top-10 most relevant candidate records for each Source 1 business.", table_cell_style),
            Paragraph("Reduces 17 Trillion possible pairs down to ~10 million high-quality pairs in seconds.", table_cell_style)
        ],
        [
            Paragraph("<b>2. Feature Engineering</b>", table_cell_bold),
            Paragraph("Calculates 19 smart similarity metrics for each candidate pair: string edit distance, token overlap, digit matching (PIN/street numbers), and group ranks.", table_cell_style),
            Paragraph("Converts raw noisy text into rich numerical signals that machine learning algorithms can understand.", table_cell_style)
        ],
        [
            Paragraph("<b>3. LightGBM ML Matching</b>", table_cell_bold),
            Paragraph("A Gradient Boosted Decision Tree model evaluates all 19 features and predicts the exact probability that two records are the same real-world company.", table_cell_style),
            Paragraph("Handles non-linear relationships, noise, and complex patterns with high accuracy and blazing speed.", table_cell_style)
        ],
        [
            Paragraph("<b>4. Macro F0.5 Thresholding</b>", table_cell_bold),
            Paragraph("Applies a mathematically tuned probability cut-off (0.94) to eliminate false merges and strictly prioritize precision.", table_cell_style),
            Paragraph("The competition metric (F0.5) penalizes false merges 2x more than missed matches.", table_cell_style)
        ]
    ]
    
    t_pipe = Table(pipeline_data, colWidths=[110, 240, 180])
    t_pipe.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EBF8FF')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_pipe)
    elements.append(Spacer(1, 10))
    
    # 3. Technologies Used & Simplified Explanation
    elements.append(Paragraph("3. Technologies Used & How Each Works (Simple Explanation)", h1_style))
    
    tech_data = [
        [
            Paragraph("<b>Technology</b>", table_cell_bold),
            Paragraph("<b>Role in the Project</b>", table_cell_bold),
            Paragraph("<b>Simple Analogy / Explanation</b>", table_cell_bold)
        ],
        [
            Paragraph("<b>LightGBM</b><br/>(Gradient Boosting)", table_cell_bold),
            Paragraph("The core AI matching brain. Trained on 25,000 ground truth entities with 5-fold cross-validation.", table_cell_style),
            Paragraph("Like a panel of 100 fast expert inspectors voting together on whether two business profiles match.", table_cell_style)
        ],
        [
            Paragraph("<b>RapidFuzz</b><br/>(C++ String Matching)", table_cell_bold),
            Paragraph("Computes Levenshtein edit distance, Token Sort Ratio, and Partial Token Set Ratio between names and addresses.", table_cell_style),
            Paragraph("Finds matches even with spelling typos ('Walmart Inc' vs 'Wal-Mart Corp') and rearranged words.", table_cell_style)
        ],
        [
            Paragraph("<b>Inverted Index & Token Hashing</b>", table_cell_bold),
            Paragraph("Custom indexing table mapping keywords to entity IDs with country-level partitioning.", table_cell_style),
            Paragraph("Like the index in the back of a textbook: instantly tells you which records mention 'Pharmacy' instead of reading the whole book.", table_cell_style)
        ],
        [
            Paragraph("<b>Scikit-Learn</b><br/>(GroupKFold CV)", table_cell_bold),
            Paragraph("Splits data into 5 validation folds grouped by <code>source1_entity_id</code> to prevent data leakage.", table_cell_style),
            Paragraph("Ensures the model is tested on brand-new companies it has never seen before, guaranteeing real-world reliability.", table_cell_style)
        ],
        [
            Paragraph("<b>Pure Python Streaming & Batching</b>", table_cell_bold),
            Paragraph("Processes 10M records chunk by chunk and directly flushes results to disk without loading everything into RAM.", table_cell_style),
            Paragraph("Like an assembly line that continuously packages items instead of trying to hold all 10 million items in your arms at once.", table_cell_style)
        ],
        [
            Paragraph("<b>Regex Normalization Engine</b>", table_cell_bold),
            Paragraph("Cleans legal suffixes ('Pvt Ltd', 'GmbH', 'SARL', 'LLC') and standardizes address formats across US, India, France.", table_cell_style),
            Paragraph("Strips away distracting legal clutter so the algorithm compares the true business identity.", table_cell_style)
        ]
    ]
    
    t_tech = Table(tech_data, colWidths=[110, 210, 210])
    t_tech.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#EDF2F7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    elements.append(t_tech)
    elements.append(Spacer(1, 10))
    
    # 4. Features & Key Results
    elements.append(Paragraph("4. Key Results & Validation Metrics", h1_style))
    elements.append(Paragraph(
        "• <b>Validation Performance:</b> Achieved a macro <b>$F_{0.5} = 0.9654$</b> at the optimal probability threshold of <b>0.94</b>.<br/>"
        "• <b>Test Coverage:</b> Successfully resolves all <b>1,732,544 Source 1 entities</b> across US, India, and France.<br/>"
        "• <b>Generated Submission Artefacts:</b><br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <code>output/matching_results.tsv</code>: Scored leaderboard submission file.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <code>output/candidate_pairs.tsv</code>: Blocking stage candidate pairs.<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;- <code>Documentation_template.md</code>: Complete technical solution report.",
        body_style
    ))
    
    doc.build(elements)
    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    build_pdf("Project_Summary_Guide.pdf")
    build_pdf("c:/Users/aniru/Desktop/lll/student_resource/Project_Summary_Guide.pdf")
