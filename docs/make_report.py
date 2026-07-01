"""Generate the full project report PDF (reportlab Platypus).

Self-contained: run with the project venv python. Avoids Unicode Greek/math glyphs
(the core PDF fonts lack them); uses ASCII spellings (delta, lambda, kappa, ~, <=, ->).
"""
import os

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image,
    Preformatted, PageBreak, HRFlowable, KeepTogether,
)

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "..", "artifacts")
OUT = os.path.join(HERE, "EGE_report.pdf")

INK = colors.HexColor("#1a1a2e")
ACCENT = colors.HexColor("#0b5394")
RED = colors.HexColor("#a61c00")
GREEN = colors.HexColor("#274e13")
PANEL = colors.HexColor("#eef2f7")
CODEBG = colors.HexColor("#f4f4f6")
RULE = colors.HexColor("#cfd8e3")

ss = getSampleStyleSheet()
S = {}
S["title"] = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=25, leading=29, textColor=INK, spaceAfter=4)
S["subtitle"] = ParagraphStyle("subtitle", parent=ss["Normal"], fontName="Helvetica-Oblique",
                               fontSize=12.5, leading=16, textColor=ACCENT, spaceAfter=2)
S["byline"] = ParagraphStyle("byline", parent=ss["Normal"], fontName="Helvetica",
                             fontSize=9, leading=12, textColor=colors.HexColor("#5a6472"))
S["h1"] = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold",
                         fontSize=15, leading=18, textColor=ACCENT, spaceBefore=16, spaceAfter=2)
S["h2"] = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold",
                         fontSize=11.5, leading=14, textColor=INK, spaceBefore=10, spaceAfter=2)
S["body"] = ParagraphStyle("body", parent=ss["Normal"], fontName="Helvetica",
                           fontSize=10, leading=14.5, textColor=INK, alignment=TA_JUSTIFY,
                           spaceAfter=6)
S["bullet"] = ParagraphStyle("bullet", parent=S["body"], leftIndent=14, bulletIndent=2,
                             spaceAfter=3)
S["code"] = ParagraphStyle("code", parent=ss["Code"], fontName="Courier", fontSize=8,
                           leading=10.5, textColor=INK)
S["callout"] = ParagraphStyle("callout", parent=S["body"], fontSize=9.5, leading=13.5,
                              alignment=TA_LEFT, spaceAfter=0, textColor=INK)
S["caption"] = ParagraphStyle("caption", parent=ss["Normal"], fontName="Helvetica-Oblique",
                              fontSize=8.5, leading=11, textColor=colors.HexColor("#5a6472"),
                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=10)
S["small"] = ParagraphStyle("small", parent=S["body"], fontSize=8.5, leading=11,
                            textColor=colors.HexColor("#5a6472"))


def P(t, st="body"):
    return Paragraph(t, S[st])


def bullets(items):
    return [Paragraph(f"<bullet>&bull;</bullet>&nbsp; {t}", S["bullet"]) for t in items]


def rule():
    return HRFlowable(width="100%", thickness=0.7, color=RULE, spaceBefore=2, spaceAfter=8)


def callout(title, text, accent=ACCENT):
    inner = [Paragraph(f"<b>{title}</b>", S["callout"]), Spacer(1, 3),
             Paragraph(text, S["callout"])]
    t = Table([[inner]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), PANEL),
        ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBEFORE", (0, 0), (0, -1), 3, accent),
    ]))
    return t


def code(text):
    p = Preformatted(text, S["code"])
    t = Table([[p]], colWidths=[165 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), CODEBG),
        ("BOX", (0, 0), (-1, -1), 0.5, RULE),
        ("LEFTPADDING", (0, 0), (-1, -1), 8), ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return t


def figure(name, caption, width_mm=158):
    path = os.path.join(ART, name)
    img = Image(path, width=width_mm * mm, height=width_mm * mm * (8.5 / 10.0))
    return KeepTogether([img, P(caption, "caption")])


def story():
    s = []
    # ---- title block ----
    s += [Spacer(1, 6),
          P("The Emergent Geometry Engine", "title"),
          P("A substrate-agnostic curvature instrument &mdash; build log, certification, "
            "and an honest negative result", "subtitle"),
          Spacer(1, 6),
          P("Technical report &nbsp;&middot;&nbsp; built under a phase-gated, "
            "generator&ne;grader discipline &nbsp;&middot;&nbsp; Python / Windows", "byline"),
          Spacer(1, 10), rule()]

    s += [callout("In one paragraph",
        "EGE turns any correlation/similarity matrix into a graph and measures the "
        "<i>shape</i> of that graph &mdash; discrete curvature, hyperbolicity, dimension "
        "&mdash; using those quantities as early-warning signals for when the underlying "
        "system reorganizes. The bet is reuse: one kernel serving two unrelated substrates "
        "(a quantum measurement transition; a neural network &lsquo;grokking&rsquo;). "
        "Phase 0 (the kernel + an independent verifier) is built and certified to machine "
        "precision. Phase 4 (a clean head-to-head against a published competitor on the "
        "grokking task) returned a <b>rigorous null</b>: graph curvature is a useful "
        "<i>description</i> of the reorganization, but it does <b>not</b> predict it earlier "
        "than the existing spectral signal. The interesting part is the machinery that made "
        "that honest &lsquo;no&rsquo; trustworthy.")]
    s += [Spacer(1, 8)]

    # ---- 1 ----
    s += [P("1 &nbsp; What it is, and the core bet", "h1"), rule()]
    s += [P("The Emergent Geometry Engine takes any symmetric correlation or similarity "
            "matrix, builds a k-nearest-neighbour graph from it, and computes the discrete "
            "geometry of that graph &mdash; Ollivier / Lin&ndash;Lu&ndash;Yau and "
            "Forman&ndash;Ricci curvature, Gromov delta-hyperbolicity, and spectral "
            "dimension. Those geometric quantities are then treated as <i>order "
            "parameters</i>: signals that move when the system producing the matrix is "
            "about to reorganize.")]
    s += [P("Why graph curvature is a sensible readout: any N&times;N relatedness matrix is "
            "a weighted graph, and graphs carry a discrete notion of curvature that is, "
            "intuitively, <i>local connectivity</i>. It is positive where neighbours form "
            "triangles (sphere-like clusters) and strongly negative on bottleneck edges that "
            "bridge otherwise-distant regions (saddle-like). That &lsquo;negative curvature "
            "lives on bottlenecks&rsquo; fact is a theorem &mdash; it underlies "
            "<i>over-squashing</i> in graph neural networks &mdash; so curvature is a cheap, "
            "local readout of how a representation is organized, and it should deflect when "
            "the representation reorganizes.")]
    s += [P("<b>Honest posture, carried throughout.</b> This is one shared mathematical "
            "object applied to two substrates &mdash; an analogy, not a grand unified "
            "theory. Calibrated claims only; null results are reportable and were reported.")]

    # ---- 2 ----
    s += [P("2 &nbsp; The engineering discipline", "h1"), rule()]
    s += [P("Four non-negotiable rules governed everything. For a programmer this is the "
            "load-bearing part:")]
    s += bullets([
        "<b>Generator &ne; grader.</b> The code that <i>computes</i> a quantity is never the "
        "code that <i>certifies</i> it. The verifier lives on a separate code path and is "
        "forbidden from importing the kernel &mdash; a structural guarantee, not a matter of "
        "trust. A bug cannot hide in a shared helper because there is no shared helper.",
        "<b>Evidence before claims.</b> Nothing asserts &lsquo;it works&rsquo;; every check "
        "prints its actual number.",
        "<b>Phase-gated.</b> Each phase has an acceptance gate with hard numeric thresholds; "
        "no advancing until it passes.",
        "<b>Test-first.</b> Classic red&ndash;green TDD &mdash; watch the test fail for the "
        "<i>right</i> reason before making it pass.",
    ])

    # ---- 3 ----
    s += [P("3 &nbsp; How it was built &mdash; the orchestration", "h1"), rule()]
    s += [P("The project was constructed by an AI coding agent acting as an orchestrator "
            "that dispatched specialised sub-agents, with the anti-self-deception machinery "
            "wired in on purpose. The repeated pattern:")]
    s += bullets([
        "<b>Research first, in isolation.</b> Before any code, separate agents verified the "
        "load-bearing claims against primary sources &mdash; exact physics formulas, library "
        "APIs, install reality on this exact machine. (This paid for itself; see Section 5.)",
        "<b>Test-first core.</b> The high-stakes verifier was authored red&ndash;green by the "
        "orchestrator, watched fail then pass on every module.",
        "<b>Parallel builders against fixed tests.</b> For independent modules the tests were "
        "written first (the spec), then builder agents were told to make them pass <i>without "
        "touching the tests</i>. A builder cannot weaken a grader it is not allowed to edit.",
        "<b>Two isolated graders at the end.</b> A read-only auditor re-ran everything and "
        "grepped for cheating (skipped tests, loosened tolerances, the verifier secretly "
        "importing the kernel). A <i>blind adversarial re-deriver</i>, forbidden from reading "
        "the source, reproduced the headline numbers with its own code. Both returned "
        "CERTIFIED.",
    ])

    # ---- 4 ----
    s += [PageBreak(), P("4 &nbsp; Phase 0 &mdash; the kernel and its verifier", "h1"), rule()]
    s += [P("Phase 0 is the foundation every later phase depends on. The centrepiece is a "
            "quantum correctness benchmark: the entanglement entropy of a 1D free-fermion "
            "chain, computed <b>four mathematically independent ways</b> and required to "
            "agree to better than 1e-9:")]
    s += bullets([
        "Peschel&rsquo;s correlation-matrix method (single-particle spectrum),",
        "brute-force exact diagonalization of the many-body Hamiltonian,",
        "the QuTiP library (independent third-party implementation),",
        "a 40-digit literature value.",
    ])
    s += [P("They agree to ~1e-15 &mdash; machine precision. Fifteen-digit agreement between "
            "four independent derivations is effectively impossible to fake with a bug; this "
            "is the epistemic bedrock the whole project stands on. The kernel also proves a "
            "real discrete-geometry theorem on actual graphs &mdash; the discrete "
            "Lichnerowicz / Lin&ndash;Lu&ndash;Yau bound, that a Ricci-curvature lower bound "
            "forces a spectral-gap lower bound (lambda_1 &gt;= kappa_min) &mdash; with a "
            "<i>negative-control</i> test that deliberately uses the wrong Laplacian to "
            "confirm the test would catch the single most common implementation error.")]
    s += [Spacer(1, 2), code(
        "EMERGENT GEOMETRY ENGINE -- PHASE 0 ACCEPTANCE GATE\n"
        "  [PASS] free-fermion: Peschel vs anchor          |S - anchor|       = 6.66e-16\n"
        "  [PASS] free-fermion: Peschel vs independent ED  |S_Peschel - S_ED| = 1.55e-15\n"
        "  [PASS] free-fermion: Peschel vs QuTiP           |Peschel - QuTiP|  = 2.11e-15\n"
        "  [PASS] inequality: trace=1 / rho>=0 / S>=0 / subadditivity / SSA / Araki-Lieb\n"
        "  [PASS] spectral_gap: K5_tight  lambda_1 = 1.250000000 == kappa_min  (tight)\n"
        "  [PASS] spectral_gap: Q3_tight  lambda_1 = 0.666666667 == kappa_min  (tight)\n"
        "  [PASS] robustness: 6 configs; ring spectral_dim mean=0.982 cov=0.075\n"
        "PHASE 0 GATE: PASS  (13 passed, 0 failed)")]
    s += [Spacer(1, 4), P("<b>Outcome:</b> 71 tests at Phase 0 (84 once Phase 4 modules were "
            "added), gate 13/13, both isolated graders CERTIFIED.", "body")]

    # ---- 5 ----
    s += [P("5 &nbsp; Three landmines the research caught", "h1"), rule()]
    s += [P("Verifying before trusting is not ceremony. Three issues, none visible from "
            "reading code:")]
    s += bullets([
        "<b>A stale dependency pin.</b> The brief required numpy&lt;2.0; on this stack that "
        "pin would have <i>broken</i> scipy/scikit-learn. Removed &mdash; everything runs on "
        "numpy 2.5.",
        "<b>A citation pointing to the wrong paper.</b> arXiv:2605.16325, leaned on for the "
        "project&rsquo;s AI-safety framing, resolves to an unrelated chemistry paper. A "
        "reviewer following it would have found nothing relevant. The correct reference "
        "(arXiv:2502.17424) was substituted.",
        "<b>A library that crashes on Windows at runtime.</b> The standard graph-curvature "
        "package hard-codes a fork() multiprocessing call that does not exist on Windows; the "
        "install succeeds, then it dies mid-run. Curvature was re-implemented from scratch via "
        "optimal transport to avoid it.",
    ])

    # ---- 6 ----
    s += [PageBreak(), P("6 &nbsp; Phase 4 &mdash; the grokking head-to-head", "h1"), rule()]
    s += [P("6.1 &nbsp; The testbed", "h2")]
    s += [P("Train a small one-layer transformer on modular addition, (a+b) mod 59, until it "
            "<i>groks</i>. Grokking is a real, well-documented oddity: the network memorizes "
            "the training set within ~100 steps (100% train accuracy, near-chance on held-out "
            "data), then &mdash; thousands of steps later, across a long dead plateau &mdash; "
            "abruptly generalizes to ~100% test accuracy. Delayed generalization with a wide "
            "gap in between, which is exactly what makes the &lsquo;does anything fire early?&rsquo; "
            "question answerable.")]
    s += [P("6.2 &nbsp; The competitor, defined correctly", "h2")]
    s += [P("The contest: does the graph-curvature signal detect the reorganization "
            "<i>earlier</i> than a published spectral early-warning signal &mdash; the "
            "2RDM spectral heat capacity? A research pass caught that the obvious guess at "
            "its definition was wrong. It is not a thermodynamic quantity; it is the variance "
            "of the eigenvalues of the covariance of <i>per-sample losses</i> over a sliding "
            "window of training checkpoints. Building the wrong guess would have produced a "
            "strawman competitor and a worthless comparison &mdash; so the competitor here is "
            "the genuine, verified signal.")]
    s += [P("6.3 &nbsp; The result (5-seed ensemble, wide-gap regime)", "h2")]
    s += [P("Across five random seeds the outcome is a clean null for the "
            "&lsquo;curvature deflects first&rsquo; thesis. A positive &lsquo;lead&rsquo; "
            "means the signal fires <i>before</i> the comparison event.")]

    tbl_data = [
        ["Metric", "Median", "[min, max]", "Across seeds"],
        ["Grok step (test acc &gt; 0.9)", "13,700", "[9,100, 17,050]", "4/5 grokked; 1 stalled"],
        ["Spectral (SHC) peak step", "2,000", "[1,950, 3,500]", "consistent, all 5"],
        ["SHC lead over grok", "+11,700", "[7,150, 15,050]", "positive in all 4 grokked"],
        ["Geometry lead over grok", "-250", "[-5,900, +1,700]", "straddles zero"],
        ["Geometry lead over SHC", "-14,450", "[-18,450, -6,400]", "negative in ALL 5"],
    ]
    tbl_rows = [[Paragraph(c, S["small"]) for c in row] for row in tbl_data]
    t = Table(tbl_rows, colWidths=[52 * mm, 24 * mm, 36 * mm, 53 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PANEL]),
        ("LINEBELOW", (0, 0), (-1, 0), 0.6, ACCENT),
        ("GRID", (0, 0), (-1, -1), 0.3, RULE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
    ]))
    # recolor the two decisive rows
    t.setStyle(TableStyle([("TEXTCOLOR", (1, 3), (1, 3), GREEN),
                           ("TEXTCOLOR", (1, 5), (1, 5), RED)]))
    s += [t, Spacer(1, 8)]
    s += [P("The spectral signal fires at step ~2,000 in every seed and leads grokking by a "
            "median <b>+11,700 steps</b>. The graph-curvature signal is roughly concurrent "
            "with generalization (it reorganizes during the <i>cleanup</i> phase) and lags "
            "the spectral signal by a median <b>~14,500 steps &mdash; negative in all five "
            "seeds, unanimous</b>. One seed did not grok within the step budget (a known "
            "seed-variance of grok timing); it is reported, not dropped.")]
    s += [figure("grokking_ensemble_frac030.png",
                 "Figure 1. Five-seed ensemble. Green = the spectral competitor (2RDM SHC); "
                 "red = graph geometry, with across-seed bands. The spectral signal spikes "
                 "early and consistently; the geometric signal is late and high-variance.")]

    s += [P("6.4 &nbsp; The false positive that got caught", "h2")]
    s += [callout("The moment the pipeline policed itself",
        "The first automated summary reported that geometry led grokking by <b>11,850 "
        "steps</b> &mdash; a headline-grade win. It was wrong. A single noisy blip in the "
        "signal had fooled the knee detector. Inspecting the raw trajectory exposed it, and "
        "the detector was rewritten to require a <i>durable</i> deflection (a commit point, "
        "not a first crossing). The pipeline converted a fake win into an honest loss. That, "
        "plus reporting the stalled seed instead of hiding it, is the entire ethos.", RED)]

    # ---- 7 ----
    s += [P("7 &nbsp; What the result means", "h1"), rule()]
    s += [P("A rigorous negative result &mdash; and the good outcome here. Three things make "
            "it trustworthy rather than merely disappointing: the competitor is the correct, "
            "<i>verified</i> signal (not a strawman, so its winning means something); the "
            "analysis caught and killed its own false positive; and the inconvenient details "
            "(the stalled seed) were reported, not swept away. Graph curvature is a useful "
            "mechanistic <i>description</i> of what grokking does to a representation &mdash; "
            "it is not an <i>earlier predictor</i> than the existing spectral signal. The "
            "project&rsquo;s entire architecture exists to make exactly this kind of "
            "calibrated &lsquo;no&rsquo; possible: a rigorous &lsquo;no&rsquo; is worth more "
            "than a flattering &lsquo;yes&rsquo; you cannot trust.")]

    # ---- 8 ----
    s += [P("8 &nbsp; Reproducibility and stack", "h1"), rule()]
    s += [P("Python 3.13 (Windows), numpy 2.5, scipy 1.18, networkx 3.6, scikit-learn, POT, "
            "GraphRicciCurvature, QuTiP 5.3 (optional cross-check), PyTorch 2.12 (CPU). 84 "
            "library tests. Everything reproduces from the command line:")]
    s += [code(
        ".venv/Scripts/python.exe -m pytest -q                       # 84 passed\n"
        ".venv/Scripts/python.exe phase0_gate.py                     # PHASE 0 GATE: PASS (13/13)\n"
        ".venv/Scripts/python.exe -m experiments.grokking.run        # train + head-to-head\n"
        ".venv/Scripts/python.exe -m experiments.grokking.ensemble_aggregate")]
    s += [Spacer(1, 6), P("Repository layout (kernel-first; the verifier never imports the "
            "kernel):", "body")]
    s += [code(
        "geometry/    kernel: kNN build + shortcut prune, curvature, delta, spectral dim, MDS\n"
        "verify/      verifier  (separate code path; never imports geometry/)\n"
        "adapters/    Adapter N (neural CKA / cosine);  Adapter Q (quantum) to come\n"
        "analysis/    early-warning signals (2RDM-SHC, geometric order parameters)\n"
        "experiments/grokking/   transformer + head-to-head runner + plots\n"
        "tests/       84 tests          docs/   reports          phase0_gate.py")]
    s += [Spacer(1, 8), rule(),
          P("Phases 1&ndash;3 (an audiovisual quantum demo and the measurement-transition "
            "sweep) and Phase 5 (an AI-safety target) remain. The certified kernel they all "
            "depend on, and one decisive &mdash; if negative &mdash; experiment on top of it, "
            "are done.", "small")]
    return s


def build():
    doc = SimpleDocTemplate(OUT, pagesize=A4,
                            leftMargin=22 * mm, rightMargin=22 * mm,
                            topMargin=18 * mm, bottomMargin=16 * mm,
                            title="The Emergent Geometry Engine", author="EGE project")

    def footer(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#9aa6b2"))
        canvas.drawString(22 * mm, 9 * mm, "The Emergent Geometry Engine")
        canvas.drawRightString(A4[0] - 22 * mm, 9 * mm, f"page {d.page}")
        canvas.restoreState()

    doc.build(story(), onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
