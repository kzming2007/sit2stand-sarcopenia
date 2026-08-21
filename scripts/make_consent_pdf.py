"""
촬영 참여 동의서 PDF 생성

    python scripts/make_consent_pdf.py --out docs/촬영참여동의서.pdf --copies 5

경진대회 사무국 안내(2026-08-21)의 조건 — "데이터 수집 및 활용 목적을
참여자에게 충분히 설명하고 **서면 동의**를 받은 경우 제한적인 기술 구현 및
모의 적용에 활용 가능" — 을 충족하기 위한 서식이다.

빈칸으로 둔 항목(지도교수, 연락처, 팀원)은 인쇄 전에 직접 채우거나
--advisor, --contact, --members 로 넘긴다.
"""

import argparse
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (KeepTogether, PageBreak, Paragraph,
                                SimpleDocTemplate, Spacer, Table, TableStyle)

FONT_DIR = r"C:\Windows\Fonts"


def register_fonts():
    """맑은 고딕 등록. 없으면 대체 폰트를 찾는다."""
    cands = [("Malgun", "malgun.ttf", "malgunbd.ttf"),
             ("NanumGothic", "NanumGothic.ttf", "NanumGothicBold.ttf"),
             ("Batang", "batang.ttc", "batang.ttc")]
    for name, reg, bold in cands:
        p1, p2 = os.path.join(FONT_DIR, reg), os.path.join(FONT_DIR, bold)
        if not os.path.exists(p1):
            continue
        pdfmetrics.registerFont(TTFont(name, p1))
        pdfmetrics.registerFont(TTFont(name + "-Bold",
                                       p2 if os.path.exists(p2) else p1))
        pdfmetrics.registerFontFamily(name, normal=name, bold=name + "-Bold")
        return name
    raise RuntimeError("한글 폰트를 찾지 못했다")


def build(out, advisor, contact, members, keep_until, copies):
    F = register_fonts()
    B = F + "-Bold"

    title = ParagraphStyle("t", fontName=B, fontSize=15, leading=19,
                           alignment=TA_CENTER, spaceAfter=2)
    sub = ParagraphStyle("s", fontName=F, fontSize=9, leading=12,
                         alignment=TA_CENTER, textColor=colors.HexColor("#555555"),
                         spaceAfter=9)
    h = ParagraphStyle("h", fontName=B, fontSize=10, leading=13,
                       spaceBefore=6, spaceAfter=2.5,
                       textColor=colors.HexColor("#1a1a1a"))
    body = ParagraphStyle("b", fontName=F, fontSize=8.9, leading=12.4,
                          spaceAfter=1)
    li = ParagraphStyle("li", parent=body, leftIndent=9, bulletIndent=2,
                        spaceAfter=0.8)
    note = ParagraphStyle("n", fontName=F, fontSize=8.1, leading=11,
                          textColor=colors.HexColor("#444444"))
    chk = ParagraphStyle("c", fontName=F, fontSize=9, leading=12.4)

    doc = SimpleDocTemplate(
        out, pagesize=A4, title="촬영 참여 동의서",
        author="2026 바이오헬스 경진대회 참가팀",
        leftMargin=19 * mm, rightMargin=19 * mm,
        topMargin=15 * mm, bottomMargin=14 * mm)

    def box(rows, widths):
        t = Table(rows, colWidths=widths)
        t.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), F),
            ("FONTSIZE", (0, 0), (-1, -1), 9.2),
            ("FONTNAME", (0, 0), (0, -1), B),
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f4f7")),
            ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#333333")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c8ccd2")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 3.5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
        ]))
        return t

    def checkbox():
        """테두리만 있는 작은 사각형. 표 안에 넣어 체크칸으로 쓴다."""
        b = Table([[""]], colWidths=[4.2 * mm], rowHeights=[4.2 * mm])
        b.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#333333")),
        ]))
        return b

    def checklist(items):
        """동의 확인 항목.

        U+2610(☐) 은 맑은 고딕에 글리프가 없어 빈칸으로 출력된다.
        폰트에 의존하지 않도록 사각형을 직접 그린다.
        """
        rows = [[checkbox(), Paragraph(s, chk)] for s in items]
        t = Table(rows, colWidths=[7 * mm, 164 * mm],
                  rowHeights=[7.2 * mm] * len(rows))
        t.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (0, -1), 1),
            ("LEFTPADDING", (1, 0), (1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return t

    story = []
    for c in range(copies):
        story += [
            Paragraph("촬영 참여 동의서", title),
            Paragraph("2026 바이오헬스 경진대회 (교내) · 스마트폰 기반 "
                      "일어서기 동작 분석 시스템 개발", sub),

            Paragraph("1. 팀 및 담당자", h),
            box([["소속", "가톨릭대학교 바이오메디컬소프트웨어학과"],
                 ["팀원", members],
                 ["지도교수", advisor],
                 ["연락처", contact]], [26 * mm, 145 * mm]),

            Paragraph("2. 목적", h),
            Paragraph(
                "스마트폰으로 촬영한 영상에서 일어서기 동작을 자동으로 분석하는 "
                "시스템을 개발하고, 그 시스템이 실제 영상에서 정상 동작하는지 "
                "확인하기 위함입니다. <b>교내 경진대회의 기술 구현 및 발표 목적에 "
                "한하며, 학술 연구나 질병 진단을 위한 것이 아닙니다.</b>", body),

            Paragraph("3. 촬영 내용", h),
            Paragraph("• 의자에 앉았다 일어서기를 5회 반복하는 것을 1회 시행으로 "
                      "하여, 총 5회 시행합니다.", li),
            Paragraph("• 스마트폰과 깊이 카메라(Azure Kinect)로 동시에 "
                      "촬영합니다.", li),
            Paragraph("• 시행 사이에 60초 이상 휴식하며, 전체 소요 시간은 약 "
                      "10~15분입니다.", li),
            Paragraph("• 언제든 중단할 수 있으며, 불편하시면 즉시 알려주십시오.", li),

            Paragraph("4. 수집하는 정보", h),
            box([["영상", "동작 영상 (얼굴이 포함됩니다)"],
                 ["신체 정보", "신장, 체중, 연령, 성별"],
                 ["촬영 조건", "의자 높이, 카메라 높이 및 거리"]],
                [26 * mm, 145 * mm]),
            Spacer(1, 3),
            Paragraph("※ 촬영되는 영상에는 <b>얼굴이 그대로 담깁니다.</b> 이 점을 "
                      "확인하신 뒤 동의해 주십시오.", note),

            Paragraph("5. 활용 범위와 보호 조치", h),
            Paragraph("• 분석에는 영상에서 추출한 <b>골격 좌표값(수치)</b>만 "
                      "사용합니다.", li),
            Paragraph("• 원본 영상은 담당자의 개인 저장장치에만 보관하며, "
                      "온라인 업로드나 외부 공유를 하지 않습니다.", li),
            Paragraph("• 제출 보고서와 발표자료에 <b>원본 영상 및 개인을 알아볼 수 "
                      "있는 화면을 포함하지 않습니다.</b>", li),
            Paragraph("• 본 대회 외의 목적(논문 발표, 정식 연구, 추가 데이터 수집)"
                      "으로 사용하려는 경우에는 <b>별도로 다시 동의를 받고 "
                      "기관생명윤리위원회(IRB)의 승인 또는 심의면제 여부를 "
                      "확인</b>한 뒤 진행합니다.", li),

            Paragraph("6. 보관 및 폐기", h),
            Paragraph(f"수집한 영상과 정보는 <b>{keep_until}</b>까지 보관하며, "
                      "기간이 지나면 원본 영상을 삭제합니다.", body),

            Paragraph("7. 동의 철회", h),
            Paragraph("언제든지 사유를 밝히지 않고 동의를 철회하실 수 있으며, "
                      "철회로 인한 어떠한 불이익도 없습니다. 철회 의사를 밝히시면 "
                      "해당 영상과 그로부터 만들어진 데이터를 즉시 폐기합니다.", body),

            Paragraph("8. 동의 확인", h),
            Paragraph("아래 항목을 읽으시고 동의하시는 경우 각 칸에 표시해 "
                      "주십시오.", note),
            Spacer(1, 2),
            checklist([
                "위 1~7의 내용을 설명 듣고 이해하였습니다.",
                "<b>얼굴이 포함된</b> 동작 영상의 촬영에 동의합니다.",
                "신장·체중·연령·성별 정보의 수집에 동의합니다.",
                "위 5항의 활용 범위 안에서 자료가 사용되는 것에 동의합니다.",
            ]),
            Spacer(1, 8),

            KeepTogether([
                box([["참여자 성명", "", "서명", "", "날짜", "20     .     .     ."],
                     ["설명한 사람", "", "서명", "", "날짜", "20     .     .     ."]],
                    [24 * mm, 40 * mm, 14 * mm, 32 * mm, 14 * mm, 47 * mm]),
                Spacer(1, 4),
                Paragraph("본 동의서는 2부를 작성하여 참여자와 담당자가 각각 "
                          "1부씩 보관합니다.", note),
            ]),
        ]
        if c < copies - 1:
            story.append(PageBreak())

    doc.build(story)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/촬영참여동의서.pdf")
    ap.add_argument("--advisor", default="")
    ap.add_argument("--contact", default="")
    ap.add_argument("--members", default="김택명 외 1명")
    ap.add_argument("--keep-until", default="2026년 12월 31일")
    ap.add_argument("--copies", type=int, default=5, help="인쇄용 매수")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    p = build(a.out, a.advisor or " " * 40, a.contact or " " * 40,
              a.members, a.keep_until, a.copies)
    print(f"생성: {p}  ({a.copies}장)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
