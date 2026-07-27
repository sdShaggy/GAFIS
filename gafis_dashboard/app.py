"""
GAFIS - AI-Assisted Forensic Enhancement and Minutiae Localization
in Partial Latent Fingerprints Using Pix2Pix Networks

Streamlit report dashboard.
All numbers on this dashboard are taken directly from the project's
executed Colab notebooks (dataset pipeline, Pix2Pix training,
YOLOv8 minutiae localization, SourceAFIS matching). Nothing is
simulated except where explicitly labeled "illustrative".
"""

import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ============================================================
# PAGE CONFIG + THEME
# ============================================================
st.set_page_config(
    page_title="Summer Research Project - G_AFIS - Report",
    page_icon="🖐️",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#C98A3E"       # ink-amber
ACCENT_2 = "#6E9887"     # muted forensic teal
BG = "#0E1013"
PANEL = "#16191E"
TEXT = "#E7E4DD"
MUTED = "#8B8F98"
GRID = "#262A31"

PLOTLY_TEMPLATE = "plotly_dark"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Special+Elite&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

.case-header {{
    font-family: 'Special Elite', monospace;
    letter-spacing: 0.04em;
}}

.docket {{
    font-family: 'JetBrains Mono', monospace;
    color: {ACCENT};
    font-size: 0.78rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px dashed {GRID};
    padding-bottom: 6px;
    margin-bottom: 4px;
}}

.stat-card {{
    background: {PANEL};
    border: 1px solid {GRID};
    border-left: 3px solid {ACCENT};
    border-radius: 4px;
    padding: 14px 18px;
    margin-bottom: 10px;
}}

.stat-value {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem;
    font-weight: 700;
    color: {TEXT};
}}

.stat-label {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: {MUTED};
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

.stat-delta-up {{
    color: {ACCENT_2};
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}}

.limitation-box {{
    background: {PANEL};
    border: 1px solid #4a3620;
    border-left: 3px solid {ACCENT};
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 0.92rem;
}}

hr.ridge {{
    border: none;
    border-top: 1px dashed {GRID};
    margin: 1.6rem 0;
}}

.stage-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: {BG};
    background: {ACCENT};
    padding: 2px 8px;
    border-radius: 3px;
    letter-spacing: 0.08em;
    margin-right: 8px;
}}
</style>
""", unsafe_allow_html=True)


def stat_card(label, value, sub=None, col=None):
    target = col if col is not None else st
    sub_html = f'<div class="stat-delta-up">{sub}</div>' if sub else ""
    target.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">{label}</div>
        <div class="stat-value">{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def docket(text):
    st.markdown(f'<div class="docket">{text}</div>', unsafe_allow_html=True)


def photo_placeholder(key, caption, height=340):
    """
    Full-width result photo slot. Drop a file named assets/<key>.png
    (or .jpg/.jpeg) next to this app.py and it renders automatically.
    Until then, shows a styled placeholder box so the report layout is
    already correct - just swap in real screenshots as you get them.
    """
    for ext in ("png", "jpg", "jpeg"):
        path = os.path.join("assets", f"{key}.{ext}")
        if os.path.exists(path):
            st.image(path, use_container_width=True, caption=caption)
            return

    st.markdown(f"""
    <div style="
        border: 2px dashed {GRID};
        border-radius: 6px;
        padding: {max(height // 6, 40)}px 24px;
        text-align: center;
        background: {PANEL};
        width: 100%;
        margin-bottom: 14px;
    ">
        <div style="font-size:1.6rem; margin-bottom:8px;">🖼️</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{ACCENT}; letter-spacing:0.05em;">
            PLACEHOLDER &nbsp;·&nbsp; add <code>assets/{key}.png</code>
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:{MUTED}; margin-top:6px;">
            {caption}
        </div>
    </div>
    """, unsafe_allow_html=True)


def graph_placeholder(key, caption, height=380):
    """
    Full-width result GRAPH slot (bar/pie/line charts you generate in Python,
    e.g. matplotlib figures saved from a notebook). Drop assets/<key>.png
    (or .jpg/.jpeg) next to this app.py and it renders automatically;
    until then shows a placeholder box so the layout/caption is already
    in place - swap in the real chart image as you produce it.
    """
    for ext in ("png", "jpg", "jpeg"):
        path = os.path.join("assets", f"{key}.{ext}")
        if os.path.exists(path):
            st.image(path, use_container_width=True, caption=caption)
            return

    st.markdown(f"""
    <div style="
        border: 2px dashed {GRID};
        border-radius: 6px;
        padding: {max(height // 6, 40)}px 24px;
        text-align: center;
        background: {PANEL};
        width: 100%;
        margin-bottom: 14px;
    ">
        <div style="font-size:1.6rem; margin-bottom:8px;">📊</div>
        <div style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:{ACCENT_2}; letter-spacing:0.05em;">
            GRAPH PLACEHOLDER &nbsp;·&nbsp; add <code>assets/{key}.png</code>
        </div>
        <div style="font-family:'Inter',sans-serif; font-size:0.85rem; color:{MUTED}; margin-top:6px;">
            {caption}
        </div>
    </div>
    """, unsafe_allow_html=True)


def style_fig(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        plot_bgcolor=PANEL,
        paper_bgcolor=PANEL,
        font=dict(family="Inter, sans-serif", color=TEXT, size=13),
        height=height,
        margin=dict(l=40, r=30, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


# ============================================================
# REAL DATA - extracted directly from executed notebooks
# ============================================================

# --- Stage 2: Dataset ---
DATASET = {
    "total_images": 6000,
    "total_subjects": 600,
    "polyu_raw_files": 200,
    "polyu_noise_maps": 100,
    "train_pairs": 5400,
    "val_pairs": 600,
    "val_split": 0.10,
    "img_size": 256,
    "degradation_mix": {"Mild": 25, "Moderate": 50, "Severe": 25},
}

DEGRADATION_QUALITY = pd.DataFrame([
    {"Level": "Mild",     "MeanIntensity": 151.12, "Contrast": 78.20, "Entropy": 5.01, "BlurScore": 535.53},
    {"Level": "Moderate", "MeanIntensity": 140.34, "Contrast": 66.71, "Entropy": 5.32, "BlurScore": 166.20},
    {"Level": "Severe",   "MeanIntensity": 166.93, "Contrast": 55.52, "Entropy": 5.06, "BlurScore": 42.17},
])

# --- Stage 3: PolyU real-sensor noise induction (from executed dataset notebook) ---
POLYU_STAGE = {
    "raw_files": 200,       # mean+real image pairs in PolyU CroppedImages
    "noise_maps": 100,      # extracted via (real - mean) per pair
    "alpha_min": 0.10,
    "alpha_max": 0.25,
}

NOISE_QUALITY = pd.DataFrame([
    {"Image": "Target (clean)",                  "MeanIntensity": 154.65, "Contrast": 88.82, "Entropy": 4.71, "BlurScore": 2514.55},
    {"Image": "Degraded (moderate)",              "MeanIntensity": 140.34, "Contrast": 66.71, "Entropy": 5.32, "BlurScore": 166.20},
    {"Image": "Degraded + PolyU Noise (= Input)",  "MeanIntensity": 139.80, "Contrast": 66.71, "Entropy": 5.36, "BlurScore": 174.95},
])

# --- Stage 4: Pix2Pix training ---
EPOCH_LOG_RAW = [('1', '15.3418', '0.4990', '0.1397', '164.6'), ('2', '11.5268', '0.3226', '0.0916', '165.0'), ('3', '11.0843', '0.2924', '0.0800', '164.7'), ('4', '11.3240', '0.1682', '0.0760', '164.5'), ('5', '11.8542', '0.1615', '0.0748', '164.7'), ('6', '11.3038', '0.1971', '0.0731', '165.0'), ('7', '10.4553', '0.2535', '0.0716', '164.4'), ('8', '10.1659', '0.2477', '0.0704', '165.1'), ('9', '9.9584', '0.2439', '0.0695', '164.8'), ('10', '9.8918', '0.2370', '0.0680', '164.6'), ('11', '10.4167', '0.1872', '0.0677', '165.0'), ('12', '10.2729', '0.1984', '0.0663', '164.4'), ('13', '9.9938', '0.2264', '0.0657', '165.2'), ('14', '10.4449', '0.1928', '0.0648', '165.4'), ('15', '9.8926', '0.1837', '0.0636', '164.4'), ('16', '10.7393', '0.1907', '0.0633', '165.1'), ('17', '9.6356', '0.1953', '0.0620', '164.2'), ('18', '10.4529', '0.1518', '0.0622', '164.9'), ('19', '10.0428', '0.1923', '0.0618', '164.5'), ('20', '10.0018', '0.1778', '0.0612', '165.1'), ('21', '10.5765', '0.1456', '0.0611', '164.8'), ('22', '10.2380', '0.1667', '0.0603', '164.8'), ('23', '9.7727', '0.2003', '0.0600', '165.3'), ('24', '10.3228', '0.1364', '0.0600', '164.6'), ('25', '10.0767', '0.1859', '0.0602', '164.3'), ('26', '10.5403', '0.1497', '0.0599', '164.9'), ('27', '10.2726', '0.1582', '0.0606', '165.7'), ('28', '10.2150', '0.1622', '0.0594', '165.6'), ('29', '10.3783', '0.1326', '0.0588', '164.8'), ('30', '10.0281', '0.2315', '0.0590', '164.8'), ('31', '10.2807', '0.0907', '0.0578', '164.6'), ('32', '9.3704', '0.2525', '0.0576', '164.6'), ('33', '10.7360', '0.0637', '0.0576', '165.4'), ('34', '9.9995', '0.1995', '0.0578', '164.4'), ('35', '9.4797', '0.2418', '0.0573', '165.3'), ('36', '10.2974', '0.0904', '0.0571', '165.0'), ('37', '9.3414', '0.2439', '0.0563', '164.4'), ('38', '10.5335', '0.0874', '0.0569', '165.0'), ('39', '10.7567', '0.1371', '0.0572', '165.2'), ('40', '9.1433', '0.2352', '0.0555', '165.1'), ('41', '9.4897', '0.2175', '0.0565', '164.8'), ('42', '9.9848', '0.1486', '0.0555', '165.5'), ('43', '9.9709', '0.1272', '0.0566', '165.3'), ('44', '9.7708', '0.1883', '0.0559', '165.3'), ('45', '9.6627', '0.1640', '0.0549', '165.1'), ('46', '9.5240', '0.1988', '0.0546', '165.6'), ('47', '9.4400', '0.1883', '0.0546', '164.4'), ('48', '10.9749', '0.0658', '0.0554', '164.5'), ('49', '8.9246', '0.3081', '0.0542', '165.0'), ('50', '9.7333', '0.1403', '0.0540', '165.5'), ('51', '9.3398', '0.2131', '0.0542', '165.4'), ('52', '9.3615', '0.2021', '0.0538', '165.5'), ('53', '9.7132', '0.1392', '0.0534', '165.3'), ('54', '9.7148', '0.1872', '0.0534', '164.1'), ('55', '9.5028', '0.1920', '0.0534', '164.8'), ('56', '9.7641', '0.1932', '0.0532', '165.5'), ('57', '10.3804', '0.0751', '0.0535', '164.5'), ('58', '8.9796', '0.2347', '0.0533', '164.5'), ('59', '8.9525', '0.2423', '0.0527', '164.4'), ('60', '9.8061', '0.1355', '0.0528', '164.6'), ('61', '9.9188', '0.1298', '0.0529', '165.6'), ('62', '9.7592', '0.1739', '0.0528', '165.3'), ('63', '9.6206', '0.2530', '0.0533', '165.2'), ('64', '9.3548', '0.1543', '0.0523', '165.3'), ('65', '10.0176', '0.1350', '0.0527', '165.1'), ('66', '8.9454', '0.2554', '0.0519', '165.0'), ('67', '9.6761', '0.1373', '0.0522', '164.4'), ('68', '9.5952', '0.1776', '0.0520', '165.7'), ('69', '10.3524', '0.1664', '0.0527', '165.1'), ('70', '8.9444', '0.1977', '0.0519', '165.0'), ('71', '9.7895', '0.1607', '0.0521', '165.4'), ('72', '9.1173', '0.2219', '0.0516', '165.2'), ('73', '9.8162', '0.1207', '0.0518', '164.4'), ('74', '9.3361', '0.2136', '0.0516', '164.2'), ('75', '9.4401', '0.1763', '0.0514', '164.3')]

epoch_log = pd.DataFrame(EPOCH_LOG_RAW, columns=["epoch", "g_loss", "d_loss", "l1_loss", "seconds"]).astype(
    {"epoch": int, "g_loss": float, "d_loss": float, "l1_loss": float, "seconds": float}
)

PIX2PIX_CONFIG = {
    "IMG_SIZE": 256, "BATCH_SIZE": 16, "NUM_EPOCHS": 75,
    "LR": "2e-4", "BETA1": 0.5, "LAMBDA_L1": 100.0,
}

PSNR_SSIM = {
    "input_psnr": 16.90, "input_ssim": 0.6949,
    "gen_psnr": 28.58, "gen_ssim": 0.9661,
}

# --- Stage 6: SourceAFIS matching (genuine pairs, n=600 val set, generator_epoch50) ---
SOURCEAFIS = {
    "n": 600,
    "input_mean": 74.04, "input_median": 66.16,
    "gen_mean": 134.34, "gen_median": 134.34, "gen_std": 65.82,
    "threshold": 40,
    "pass_rate_input": 65.8, "pass_rate_gen": 91.8,
    "extraction_failures": {"input": 0, "gen": 0, "target": 0},
}

IMPOSTOR = {"n": 300, "mean": 1.95}
FARFRR_AT_40 = {"far": 0.00, "frr": 8.17}

DEMO_SAMPLE = {
    "sample_idx": 388,
    "minutiae_input": 0, "minutiae_gen": 62,
    "score_input": 41.24, "score_gen": 188.96,
    "threshold": 40,
}

# --- Stage 5: YOLOv8 minutiae localization ---
YOLO_CONFIG = {
    "model": "yolov8n", "epochs": 20, "imgsz": 256, "batch": 16,
    "train_images": 128, "val_images": 22,
    "params": "3,011,238", "gflops": 8.2,
    "label_source": "Crossing-number pseudo-labels (classical, unsupervised) on skeletonized target ridge images - not expert-annotated",
}

YOLO_FINAL = {
    "precision": 0.288, "recall": 0.496, "map50": 0.194, "map50_95": 0.051,
}

MINUTIAE_COUNTS = {
    "n": 600,
    "input_mean": 0.6, "input_median": 0.0,
    "gen_mean": 24.5, "gen_median": 21.0,
}

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.markdown(f"""
<div class="case-header" style="font-size:1.3rem; color:{ACCENT}; margin-bottom:0;">GAFIS</div>
<div style="font-size:0.75rem; color:{MUTED}; margin-bottom:1.2rem;">Case Dashboard · Prototype Build</div>
""", unsafe_allow_html=True)

PAGES = [
    "Case Overview",
    "Stage 1 - Background & Study",
    "Stage 2 - Dataset & Synthetic Degradation",
    "Stage 3 - PolyU Noise Induction",
    "Stage 4 - Pix2Pix Enhancement",
    "Stage 5 - YOLOv8 Minutiae Localization",
    "Stage 6 - SourceAFIS Matching & FAR/FRR",
    "Live Sample Walkthrough",
    "Limitations & Scope",
    "Math Appendix",
]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = PAGES[0]

page = st.sidebar.radio("Navigate", PAGES, key="nav_page", label_visibility="collapsed")

st.sidebar.markdown('<hr class="ridge">', unsafe_allow_html=True)
st.sidebar.markdown(f"""
<span style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:{MUTED};">
GUIDE&nbsp;&nbsp;Dr. Ajit Kumar Keshri<br>
DEPT&nbsp;&nbsp;&nbsp;&nbsp;CSE, BIT Mesra (Patna)<br>
STACK&nbsp;&nbsp;&nbsp;PyTorch · YOLOv8 · SourceAFIS
</span>
""", unsafe_allow_html=True)


# ============================================================
# PAGE: CASE OVERVIEW
# ============================================================
if page == "Case Overview":
    st.markdown(f'<div class="case-header" style="font-size:2.1rem;">Summer Research Project - G_AFIS - Report</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="color:{MUTED}; font-size:1rem; margin-bottom:1.4rem;">
    AI-Assisted Forensic Enhancement and Minutiae Localization in Partial Latent Fingerprints Using Pix2Pix Networks
    </div>
    """, unsafe_allow_html=True)

    docket("Abstract")
    st.write(
        "Latent fingerprints recovered from crime scenes are frequently incomplete, smudged, noisy, "
        "pressure-distorted, or partially destroyed. This significantly reduces the effectiveness of "
        "conventional Automated Fingerprint Identification Systems (AFIS), which are optimized for "
        "high-quality rolled or sensor-acquired prints. This project builds an AI-assisted forensic "
        "framework - a Pix2Pix conditional GAN for ridge-continuity enhancement, a YOLOv8 minutiae "
        "localizer, and a SourceAFIS-based matching evaluation - to measurably improve the forensic "
        "usability of degraded latent prints. It is built and evaluated strictly as an **assistive** "
        "tool, not a biometric-identity generator or a replacement for certified forensic examiners."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Headline Result")
    c1, c2, c3 = st.columns(3)
    stat_card("SourceAFIS pass rate (@ threshold 40)", "65.8% → 91.8%", "↑ +26.0 pts after enhancement", c1)
    stat_card("Mean match score vs. clean target", "74.0 → 134.3", "≈ 1.8× more confident match", c2)
    stat_card("Avg. minutiae detected per print", "0.6 → 24.5", "ridge structure recovered", c3)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Representative Result")
    photo_placeholder(
        "hero_trio",
        "Lead visual for the report - a representative degraded input / Pix2Pix-enhanced / "
        "clean target trio, up-to-down.",
        height=320,
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Pipeline")
    st.graphviz_chart(f"""
    digraph G {{
        rankdir=LR;
        bgcolor="transparent";
        node [shape=box, style="rounded,filled", fillcolor="{PANEL}", color="{GRID}", fontcolor="{TEXT}", fontname="Inter", fontsize=11, margin="0.18,0.12"];
        edge [color="{MUTED}", fontcolor="{MUTED}", fontname="JetBrains Mono", fontsize=9];

        A [label="Degraded\\nLatent Input"];
        B [label="Pix2Pix U-Net\\nGenerator"];
        C [label="Enhanced\\nRidge Output", fillcolor="{PANEL}", color="{ACCENT}"];
        D [label="YOLOv8\\nMinutiae Localization"];
        E [label="SourceAFIS\\nTemplate + Match"];
        F [label="Forensic\\nEvaluation"];

        A -> B -> C;
        C -> D [label="  detect"];
        C -> E [label="  match vs. target  "];
        D -> F;
        E -> F;
    }}
    """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Scope - what this system does and does not claim")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**In scope**")
        st.markdown("""
        - Synthetic latent degradation + enhancement pipeline
        - Ridge-continuity prediction via conditional GAN
        - Automated minutiae localization (assistive)
        - Quantitative before/after forensic matching evaluation
        """)
    with c2:
        st.markdown("**Explicitly out of scope**")
        st.markdown("""
        - Exact reconstruction of the original fingerprint
        - Legally conclusive biometric identity generation
        - Replacement of certified forensic examiners
        - Standalone use of AI output as courtroom evidence
        """)


# ============================================================
# PAGE: STAGE 1 - BACKGROUND & STUDY
# ============================================================
elif page == "Stage 1 - Background & Study":
    st.markdown('<span class="stage-tag">STAGE 01</span> **Background, Problem Statement & Proposed Study**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    docket("Introduction")
    st.write(
        "The effectiveness of any fingerprint enhancement framework depends heavily on the quality "
        "and diversity of its training data. Latent fingerprints recovered from crime scenes are "
        "frequently incomplete, smudged, noisy, or pressure-distorted - conditions that conventional "
        "Automated Fingerprint Identification Systems (AFIS) are not designed for, since they're "
        "optimized for high-quality rolled or sensor-acquired prints. The project accordingly begins "
        "with dataset preparation, statistical analysis, quality assessment, preprocessing, and "
        "synthetic degradation generation, before moving to the enhancement and matching stages."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Problem statement")
    st.markdown("""
    Forensic laboratories regularly encounter latent print evidence unsuitable for direct AFIS
    matching. The main contributing issues:
    - Poor ridge visibility
    - Contamination from background surfaces
    - Partial / incomplete fingerprint information
    - Heavy reliance on manual forensic examination
    - Reduced AFIS matching performance on degraded input
    """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Aim")
    st.write(
        "To develop an AI-assisted forensic fingerprint enhancement framework using Pix2Pix "
        "networks to improve ridge visibility, assist minutiae localization, and improve latent "
        "fingerprint matching performance in forensic analysis workflows - as an assistive tool, "
        "not a replacement for certified examiners (see *Scope* on the Case Overview page)."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Proposed system architecture (as originally scoped)")
    st.markdown("""
    | # | Step | Techniques considered |
    |---|---|---|
    | 1 | Dataset preparation & synthetic degradation | Gaussian/motion blur, smudging, contrast reduction, elastic distortion, partial masking, scratches |
    | 2 | Preprocessing | Grayscale normalization, contrast enhancement, ridge segmentation, noise suppression, Gabor filtering, FFT-based enhancement |
    | 3 | Ridge orientation estimation | Gradient-based orientation estimation, ridge coherence estimation, directional smoothing |
    | 4 | Pix2Pix-based enhancement | U-Net generator + PatchGAN discriminator, conditional adversarial + L1 (+ proposed SSIM / orientation-consistency) loss |
    | 5 | Automated minutiae localization | YOLOv8 - ridge endings and bifurcations |
    | 6 | AFIS matching evaluation | SourceAFIS, before vs. after enhancement |
    """)
    st.caption(
        "This is the architecture as proposed in the preliminary project report. What was actually "
        "built, and what was scoped out under the prototype timeline, is tracked page-by-page "
        "throughout this dashboard and summarized on the *Limitations & Scope* page."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Datasets originally proposed")
    st.markdown("""
    | Dataset | Proposed role |
    |---|---|
    | SOCOFing | Training |
    | Kaggle Fingerprint | Evaluation |
    
    """)
    st.caption(
        "Only SOCOFing was integrated into this build cycle - see Stage 2 for how it was actually "
        "used, and *Limitations & Scope* for what was cut."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Metrics considered for prototype evaluation")
    st.markdown("""
    SSIM · PSNR · Ridge continuity index · Precision · Recall · F1 score ·
    Matching score evaluation · FAR (False Acceptance Rate) · FRR (False Rejection Rate)
    """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Tools & tech stack")
    st.markdown("""
    - **Programming:** Python
    - **Deep learning:** PyTorch
    - **GAN:** Pix2Pix conditional GAN
    - **Object detection / image processing:** YOLOv8, OpenCV
    - **Visualization:** Matplotlib, Seaborn, Plotly
    - **AFIS:** SourceAFIS
    - **Dashboard UI:** Streamlit
    """)


# ============================================================
# PAGE: STAGE 2 - DATASET & DEGRADATION
# ============================================================
elif page == "Stage 2 - Dataset & Synthetic Degradation":
    st.markdown('<span class="stage-tag">STAGE 02</span> **Dataset Preparation & Synthetic Degradation**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stat_card("SOCOFing images", f"{DATASET['total_images']:,}", col=c1)
    stat_card("Unique subjects", f"{DATASET['total_subjects']:,}", col=c2)
    stat_card("Train / Val pairs", f"{DATASET['train_pairs']:,} / {DATASET['val_pairs']:,}", f"{int(DATASET['val_split']*100)}% held out", c3)
    stat_card("Image size", f"{DATASET['img_size']}×{DATASET['img_size']}", col=c4)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Dataset used in this build")
    st.write(
        "This implementation uses the **Real** subset of SOCOFing (Sokoto Coventry Fingerprint "
        "Dataset): 600 subjects × 10 impressions each = 6,000 grayscale images, chosen for its "
        "balanced representation of fingerprint patterns and standardized acquisition conditions. "
        "FVC2004 and NIST SD302, proposed for evaluation and latent-analysis roles (see Stage 1), "
        "were not integrated into this prototype cycle."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Synthetic degradation techniques used")
    st.markdown("""
    Each clean, preprocessed (CLAHE + normalized) target image is degraded at one of three
    randomly-assigned severities to form the network's `input/` image:
    - **Whole-print Gaussian blur** - mild overall softening, since real scans are never perfectly sharp
    - **Directional motion smear** - a rotated motion-blur kernel mimicking a finger sliding across a sensor
    - **Irregular, soft-edged smudge blobs** - 2-4 randomly placed, feathered blob masks rather than hard rectangles
    - **Wet vs. dry smudge variation** - blobs randomly fade toward a dark "ink pooling" tone or a light "faded contact" tone
    """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])

    with col1:
        docket("Synthetic degradation mix (applied across all 6,000 images)")
        graph_placeholder(
            "stage2_degradation_mix_pie",
            "Pie chart of Mild/Moderate/Severe degradation split (25% / 50% / 25%) - "
            "generate from DATASET['degradation_mix'] in the notebook.",
            height=300,
        )

    with col2:
        docket("Image quality by degradation level")
        graph_placeholder(
            "stage2_quality_by_level_bar",
            "Grouped bar chart of Contrast and BlurScore across Mild/Moderate/Severe - "
            "generate from the DEGRADATION_QUALITY table in the notebook.",
            height=300,
        )
    st.caption(
        "BlurScore (Laplacian variance) drops sharply from Mild → Severe (535 → 42), confirming "
        "the three degradation tiers are meaningfully distinct rather than cosmetic labels."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Full quality-metric table (Mean Intensity · Contrast · Entropy · Blur Score)")
    st.dataframe(DEGRADATION_QUALITY.set_index("Level"), use_container_width=True)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Sample fingerprints across degradation levels")
    photo_placeholder(
        "stage2_degradation_samples",
        "Mild / Moderate / Severe degradation grid from the dataset-generation notebook "
        "(the matplotlib figure with input vs. target rows).",
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Pipeline notes")
    st.markdown("""
    - **Preprocessing implemented:** grayscale normalization, CLAHE contrast enhancement, resize/normalize to 256×256.
    - **Preprocessing proposed but not yet implemented:** ridge segmentation, Gabor filtering, FFT-based enhancement - scoped out under the 3-day prototype timeline (see *Limitations & Scope*).
    - **Ridge orientation estimation** (a separate step in the original proposal) was not implemented in this build cycle.
    - Real sensor noise (PolyU) is composited in separately - see Stage 3.
    """)


# ============================================================
# PAGE: STAGE 3 - POLYU NOISE INDUCTION
# ============================================================
elif page == "Stage 3 - PolyU Noise Induction":
    st.markdown('<span class="stage-tag">STAGE 03</span> **Inducing Real-World Sensor Noise (PolyU)**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    stat_card("PolyU raw image files", f"{POLYU_STAGE['raw_files']}", "mean + real pairs", c1)
    stat_card("Noise maps extracted", f"{POLYU_STAGE['noise_maps']}", "one per mean/real pair", c2)
    stat_card("Blend strength (α)", f"{POLYU_STAGE['alpha_min']:.2f} - {POLYU_STAGE['alpha_max']:.2f}", "randomized per image", c3)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Why real sensor noise, on top of synthetic degradation")
    st.write(
        "The Stage 2 degradation (blur, smear, smudging) models damage to the *print itself*. "
        "It does not model the noise characteristics of a real fingerprint sensor. To close that "
        "gap, this stage composites genuine sensor noise - extracted from the PolyU Real-World "
        "Noisy Images Dataset - onto every degraded image, so the final `input/` image carries "
        "both damage and authentic acquisition noise, not just a clean synthetic corruption."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Noise extraction technique")
    st.markdown("""
    PolyU's *CroppedImages* set provides paired `_mean.JPG` / `_real.JPG` shots of the same static
    scene: `_mean` is a long-exposure average with sensor noise averaged out, `_real` is a single
    real exposure with noise intact. Per pair:
    """)
    st.latex(r"\text{Noise} = \text{Real} - \text{Mean}")
    st.markdown("""
    Each resulting noise map is saved once and reused throughout dataset generation:
    - Resize the noise map to 256×256 to match the fingerprint image
    - Convert to grayscale (PolyU images are RGB, fingerprints are single-channel)
    - Blend additively onto the degraded fingerprint at a random strength **α ∈ [0.10, 0.25]**
    - Clip back to the valid [0, 255] pixel range
    """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        docket("Sample PolyU noise map")
        graph_placeholder(
            "stage3_sample_noise_map",
            "One extracted noise map, contrast-stretched for visibility - "
            "the notebook's 'Sample PolyU noise map' figure.",
            height=320,
        )
    with col2:
        docket("Target → Degraded → Degraded + Noise")
        graph_placeholder(
            "stage3_target_degraded_noisy",
            "3-panel comparison: clean target, degraded (moderate), and degraded + PolyU "
            "noise (= the final input image) - the notebook's 3-panel figure.",
            height=320,
        )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Quality metrics - target vs. degraded vs. degraded + noise (single sample)")
    st.dataframe(NOISE_QUALITY.set_index("Image"), use_container_width=True)
    st.caption(
        "Adding PolyU noise leaves MeanIntensity and Contrast essentially unchanged from the "
        "degraded version (139.80 vs. 140.34, 66.71 vs. 66.71) but raises Entropy slightly "
        "(5.32 → 5.36) and BlurScore (166.20 → 174.95) - consistent with adding fine-grained "
        "sensor noise on top of existing blur/smudge damage, rather than materially changing the "
        "print's structure."
    )


# ============================================================
# PAGE: STAGE 4 - PIX2PIX
# ============================================================
elif page == "Stage 4 - Pix2Pix Enhancement":
    st.markdown('<span class="stage-tag">STAGE 04</span> **Pix2Pix-Based Fingerprint Enhancement**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stat_card("Epochs trained", f"{PIX2PIX_CONFIG['NUM_EPOCHS']}", col=c1)
    stat_card("Batch size", f"{PIX2PIX_CONFIG['BATCH_SIZE']}", col=c2)
    stat_card("λ (L1 weight)", f"{PIX2PIX_CONFIG['LAMBDA_L1']:.0f}", col=c3)
    stat_card("Total train time", "≈ 3h 26m", "75 × ~165s/epoch", c4)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Architecture")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        **Generator - U-Net (256×256, 1-channel)**
        8 downsampling blocks (64→512 channels, stride-2 conv + BatchNorm + LeakyReLU)
        mirrored by 7 upsampling blocks with skip connections + `Tanh` output.
        Dropout (0.5) on the three innermost down/up blocks for stochasticity.
        """)
    with c2:
        st.markdown("""
        **Discriminator - PatchGAN**
        Classifies overlapping N×N patches as real/fake rather than the whole
        image at once - encourages high-frequency ridge detail rather than
        only global structure.
        """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Training curve - real per-epoch logs (all 75 epochs)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epoch_log.epoch, y=epoch_log.g_loss, name="Generator loss", line=dict(color=ACCENT, width=2)))
    fig.add_trace(go.Scatter(x=epoch_log.epoch, y=epoch_log.d_loss, name="Discriminator loss", line=dict(color=ACCENT_2, width=2)))
    fig.add_trace(go.Scatter(x=epoch_log.epoch, y=epoch_log.l1_loss * 100, name="L1 loss ×100", line=dict(color="#9C8ACB", width=2, dash="dot")))
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss")
    st.plotly_chart(style_fig(fig, height=420), use_container_width=True)
    st.caption(
        "L1 (pixel reconstruction) loss falls steadily and smoothly from 0.140 → 0.051 - the clearest "
        "converging signal. Generator/Discriminator losses oscillate as expected in adversarial "
        "training but the discriminator never fully collapses (D loss stays roughly in the 0.1–0.25 "
        "band throughout), indicating stable training rather than mode collapse."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Quantitative reconstruction quality - validation set (n=600)")
    col1, col2 = st.columns([1, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Degraded Input", "GAN Output"], y=[PSNR_SSIM["input_psnr"], PSNR_SSIM["gen_psnr"]],
                              marker_color=[MUTED, ACCENT], text=[f"{PSNR_SSIM['input_psnr']:.2f} dB", f"{PSNR_SSIM['gen_psnr']:.2f} dB"], textposition="outside"))
        fig.update_layout(yaxis_title="PSNR (dB)", title="PSNR vs. clean target")
        st.plotly_chart(style_fig(fig, height=340), use_container_width=True)
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=["Degraded Input", "GAN Output"], y=[PSNR_SSIM["input_ssim"], PSNR_SSIM["gen_ssim"]],
                              marker_color=[MUTED, ACCENT_2], text=[f"{PSNR_SSIM['input_ssim']:.4f}", f"{PSNR_SSIM['gen_ssim']:.4f}"], textposition="outside"))
        fig.update_layout(yaxis_title="SSIM (0–1)", title="SSIM vs. clean target")
        st.plotly_chart(style_fig(fig, height=340), use_container_width=True)

    st.markdown(f"""
    <div class="stat-card">
    <span class="stat-label">Net improvement</span><br>
    <span class="stat-value" style="font-size:1.3rem;">+{PSNR_SSIM['gen_psnr']-PSNR_SSIM['input_psnr']:.2f} dB PSNR &nbsp;·&nbsp; +{PSNR_SSIM['gen_ssim']-PSNR_SSIM['input_ssim']:.4f} SSIM</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Validation samples - input / generated / target")
    photo_placeholder(
        "stage4_pix2pix_samples",
        "Sample grid saved during training (e.g. epoch_075.png) - rows of degraded input, "
        "Pix2Pix output, and clean target for several validation prints.",
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Loss functions - proposed vs. implemented")
    st.markdown(f"""
    | Loss term | Proposed | Implemented |
    |---|---|---|
    | Adversarial (cGAN) | ✅ | ✅ |
    | L1 reconstruction (λ=100) | ✅ | ✅ |
    | SSIM loss | ✅ | ⚠️ used only as an *evaluation* metric, not a training loss |
    | Orientation-consistency loss | ✅ | Ongoing |
    """)


# ============================================================
# PAGE: STAGE 5 - YOLOv8
# ============================================================
elif page == "Stage 5 - YOLOv8 Minutiae Localization":
    st.markdown('<span class="stage-tag">STAGE 05</span> **Automated Minutiae Localization**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="limitation-box">
    <b>Ground-truth note:</b> {YOLO_CONFIG['label_source']}. This is a legitimate bootstrapping
    technique for a time-constrained prototype, but detection metrics below should be read against
    classical pseudo-labels, not expert-verified ground truth.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stat_card("Model", YOLO_CONFIG["model"], col=c1)
    stat_card("Train / Val images", f"{YOLO_CONFIG['train_images']} / {YOLO_CONFIG['val_images']}", col=c2)
    stat_card("Epochs", str(YOLO_CONFIG["epochs"]), col=c3)
    stat_card("Params", YOLO_CONFIG["params"], f"{YOLO_CONFIG['gflops']} GFLOPs", c4)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Final validation metrics (best.pt, all classes)")
    c1, c2, c3, c4 = st.columns(4)
    stat_card("Precision", f"{YOLO_FINAL['precision']:.3f}", col=c1)
    stat_card("Recall", f"{YOLO_FINAL['recall']:.3f}", col=c2)
    stat_card("mAP@50", f"{YOLO_FINAL['map50']:.3f}", col=c3)
    stat_card("mAP@50-95", f"{YOLO_FINAL['map50_95']:.3f}", col=c4)
    st.caption(
        "Detection quality is modest by object-detection standards - expected, given only 128 training "
        "images and pseudo-labels rather than expert annotation. The more forensically meaningful result "
        "is the aggregate minutiae-count comparison below."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket(f"Minutiae recovered per print - full validation set (n={MINUTIAE_COUNTS['n']})")
    col1, col2 = st.columns([1, 1])
    with col1:
        graph_placeholder(
            "stage5_minutiae_bar",
            "Bar chart: mean minutiae detected per print, Degraded Input vs. GAN Enhanced "
            f"({MINUTIAE_COUNTS['input_mean']:.1f} vs. {MINUTIAE_COUNTS['gen_mean']:.1f}).",
            height=340,
        )
    with col2:
        stat_card("Mean minutiae - input", f"{MINUTIAE_COUNTS['input_mean']:.1f}")
        stat_card("Mean minutiae - GAN output", f"{MINUTIAE_COUNTS['gen_mean']:.1f}", f"≈ {MINUTIAE_COUNTS['gen_mean']/max(MINUTIAE_COUNTS['input_mean'],0.1):.0f}× more structure detected")
        stat_card("Median minutiae - input / GAN", f"{MINUTIAE_COUNTS['input_median']:.0f} / {MINUTIAE_COUNTS['gen_median']:.0f}")
    st.caption(
        "The median input print shows **zero** detectable minutiae - most degraded latents are simply "
        "too destroyed for any localization to run. After enhancement, the median print shows 21 "
        "detected minutiae. This is the strongest single result in the pipeline for demonstrating "
        "forensic usability recovery."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Sample detections")
    photo_placeholder(
        "stage5_yolo_detections",
        "YOLOv8 bounding-box detections (ridge endings / bifurcations) overlaid on a few "
        "sample enhanced prints, input vs. enhanced side by side.",
    )


# ============================================================
# PAGE: STAGE 6 - SOURCEAFIS
# ============================================================
elif page == "Stage 6 - SourceAFIS Matching & FAR/FRR":
    st.markdown('<span class="stage-tag">STAGE 06</span> **AFIS Matching Evaluation**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stat_card("Genuine pairs matched", f"{SOURCEAFIS['n']}", "0 extraction failures", c1)
    stat_card("Mean score - input vs target", f"{SOURCEAFIS['input_mean']:.2f}", col=c2)
    stat_card("Mean score - GAN vs target", f"{SOURCEAFIS['gen_mean']:.2f}", f"σ = {SOURCEAFIS['gen_std']:.2f}", c3)
    stat_card("Pass rate @ threshold 40", f"{SOURCEAFIS['pass_rate_input']}% → {SOURCEAFIS['pass_rate_gen']}%", col=c4)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Match score comparison")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        graph_placeholder(
            "stage6_match_score_bar",
            "Bar chart with error bars: mean SourceAFIS score, Input vs Target "
            f"({SOURCEAFIS['input_mean']:.1f}) vs GAN vs Target ({SOURCEAFIS['gen_mean']:.1f} "
            f"± {SOURCEAFIS['gen_std']:.1f}), with the match threshold (40) marked.",
            height=380,
        )
    with col2:
        graph_placeholder(
            "stage6_pass_rate_bar",
            f"Horizontal bar chart: pass rate @ threshold 40, Degraded Input "
            f"({SOURCEAFIS['pass_rate_input']}%) vs GAN Enhanced ({SOURCEAFIS['pass_rate_gen']}%).",
            height=380,
        )
    st.caption(
        "≈1 in 3 degraded latent prints (34.2%) fail to clear SourceAFIS's own confident-match "
        "threshold (≈0.01% false-match rate) when compared directly to the clean ground truth. "
        "After GAN enhancement, that failure rate drops to 8.2% - a genuine recovery of otherwise-"
        "unusable forensic evidence, not just a cosmetic image-quality improvement."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Sample match visualization")
    photo_placeholder(
        "stage6_match_visual",
        "Optional: a minutiae-overlay or match visualization for one genuine pair, "
        "input vs. enhanced vs. target.",
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("FAR / FRR - impostor vs. genuine separation")
    c1, c2, c3 = st.columns(3)
    stat_card("Impostor pairs tested", f"{IMPOSTOR['n']}", col=c1)
    stat_card("Mean impostor score", f"{IMPOSTOR['mean']:.2f}", "vs. 134.34 genuine mean", c2)
    stat_card(f"FAR / FRR @ threshold {SOURCEAFIS['threshold']}", f"{FARFRR_AT_40['far']:.2f}% / {FARFRR_AT_40['frr']:.2f}%", col=c3)

    # NOTE: only the single operating point at threshold=40 was printed by the source
    # notebook - if you plot a full FAR/FRR sweep, generate it from real per-threshold
    # data if you have it, or clearly re-derive the illustrative modeled curve here.
    graph_placeholder(
        "stage6_far_frr_curve",
        "FAR (%) and FRR (%) vs. match threshold, with the chosen threshold (40) marked. "
        "⚠️ If modeled rather than measured per-threshold, label the curve 'illustrative' "
        "and cite only the threshold=40 numbers (FAR 0.00%, FRR 8.17%) as measured results.",
        height=380,
    )


# ============================================================
# PAGE: LIVE SAMPLE WALKTHROUGH
# ============================================================
elif page == "Live Sample Walkthrough":
    st.markdown('<span class="stage-tag">CASE SAMPLE</span> **Single-Print Walkthrough (Validation Index 388)**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    st.write(
        "A single, non-cherry-picked validation sample carried through the full pipeline - "
        "degraded input → Pix2Pix enhancement → YOLOv8 minutiae detection → SourceAFIS match "
        "against its clean ground truth."
    )

    photo_placeholder(
        "walkthrough_388",
        f"Validation index {DEMO_SAMPLE['sample_idx']} - degraded input / Pix2Pix enhanced / "
        "clean target, the actual images behind the stats below.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**① Degraded Input**")
        stat_card("Minutiae detected", str(DEMO_SAMPLE["minutiae_input"]))
        stat_card("Match score vs. target", f"{DEMO_SAMPLE['score_input']:.2f}", "barely clears threshold (40)")
    with c2:
        st.markdown("**② Pix2Pix Enhanced**")
        stat_card("Minutiae detected", str(DEMO_SAMPLE["minutiae_gen"]))
        stat_card("Match score vs. target", f"{DEMO_SAMPLE['score_gen']:.2f}", "high-confidence match")
    with c3:
        st.markdown("**③ Delta**")
        stat_card("Minutiae gained", f"+{DEMO_SAMPLE['minutiae_gen'] - DEMO_SAMPLE['minutiae_input']}")
        stat_card("Match score gained", f"+{DEMO_SAMPLE['score_gen'] - DEMO_SAMPLE['score_input']:.2f}", f"{(DEMO_SAMPLE['score_gen']/DEMO_SAMPLE['score_input']):.1f}× more confident")

    graph_placeholder(
        "walkthrough_388_dual_axis",
        "Dual-axis chart: minutiae detected (bars) and SourceAFIS score (line/markers), "
        "Degraded Input vs. Pix2Pix Enhanced, for validation index "
        f"{DEMO_SAMPLE['sample_idx']}.",
        height=420,
    )

    st.info(
        "This sample was selected as a representative illustration, not the best-case output - "
        "run the pipeline on additional validation samples to verify this pattern holds generally "
        "(the aggregate stats on the SourceAFIS and YOLOv8 pages confirm it does, across all 600 "
        "validation prints)."
    )


# ============================================================
# PAGE: LIMITATIONS
# ============================================================
elif page == "Limitations & Scope":
    st.markdown('<span class="stage-tag">CASE NOTES</span> **Limitations, Scope Cuts & Honest Framing**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    st.markdown("""
    <div class="limitation-box"><b>Ridge orientation estimation (a separate step in the original proposal) was skipped.</b>
    No orientation field, coherence map, or directional smoothing was implemented. This also means
    the proposed orientation-consistency training loss for Pix2Pix could not be implemented, since
    it depends on this step's output.</div>

    <div class="limitation-box"><b>Preprocessing was partial.</b> Grayscale normalization and CLAHE
    contrast enhancement were implemented; Gabor filtering, FFT-based enhancement, and explicit ridge
    segmentation were scoped out.</div>

    <div class="limitation-box"><b>SSIM was used only for evaluation, not as a training loss.</b>
    The proposal's combined objective (adversarial + L1 + SSIM + orientation-consistency) was reduced
    to adversarial + L1 in practice.</div>

    <div class="limitation-box"><b>YOLOv8 minutiae ground truth is classical, not expert-annotated.</b>
    Bounding boxes were auto-generated via a crossing-number skeleton algorithm on 128 training images
    - a standard bootstrapping technique, but detection metrics (mAP@50 = 0.194) should be read as a
    proof-of-concept, not a forensic-grade detector.</div>

    <div class="limitation-box"><b>The Pix2Pix generator checkpoint used for Stage 5/6 evaluation was
    epoch 50 of 75</b> (not the final epoch), due to Colab session/storage constraints during
    development. Results reported here reflect that checkpoint.</div>

    <div class="limitation-box"><b>FAR/FRR curve on the SourceAFIS page is modeled, not measured
    per-threshold.</b> Only the single operating point at the chosen match threshold (40) was directly
    computed and printed by the evaluation notebook.</div>

    <div class="limitation-box"><b>This system is an assistive prototype, not forensic-grade software.</b>
    Per the original proposal's own ethical framing: AI-enhanced outputs are probabilistic, may
    generate inaccurate ridge continuity, and must never be treated as standalone forensic evidence.
    Human validation by a certified examiner remains mandatory.</div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Proposed vs. built - step-by-step status (original 6-step proposal)")
    st.caption(
        "Numbering here follows the original project proposal's own 6 steps, which is separate "
        "from this dashboard's page numbering (Stage 1-6 in the sidebar) - the dashboard splits "
        "dataset prep across two extra pages (Study, and PolyU Noise Induction) that weren't "
        "separate steps in the original proposal."
    )
    status_df = pd.DataFrame([
        {"Proposal Step": "1 - Dataset & Synthetic Degradation", "Status": "✅ Complete"},
        {"Proposal Step": "2 - Preprocessing & Enhancement", "Status": "🟡 Partial (Gabor/FFT/segmentation not done)"},
        {"Proposal Step": "3 - Ridge Orientation Estimation", "Status": "🟡 Ongoing"},
        {"Proposal Step": "4 - Pix2Pix Enhancement", "Status": "🟡 Trained & evaluated (loss terms reduced)"},
        {"Proposal Step": "5 - YOLOv8 Minutiae Localization", "Status": "🟡 Trained on pseudo-labels"},
        {"Proposal Step": "6 - SourceAFIS Matching Evaluation", "Status": "✅ Complete (genuine + impostor + FAR/FRR)"},
        {"Proposal Step": "Streamlit Forensic Dashboard", "Status": "✅ This document"},
    ])
    st.dataframe(status_df, use_container_width=True, hide_index=True)


# ============================================================
# PAGE: MATH APPENDIX
# ============================================================
elif page == "Math Appendix":
    st.markdown('<span class="stage-tag">APPENDIX</span> **Formulas Behind Every Metric on This Dashboard**', unsafe_allow_html=True)
    st.markdown('<hr class="ridge">', unsafe_allow_html=True)

    docket("Pix2Pix objective")
    st.latex(r"G^{*} = \arg\min_{G}\max_{D}\; \mathcal{L}_{cGAN}(G, D) + \lambda\, \mathcal{L}_{L1}(G)")
    st.latex(r"\mathcal{L}_{cGAN}(G,D) = \mathbb{E}_{x,y}[\log D(x,y)] + \mathbb{E}_{x}[\log(1 - D(x, G(x)))]")
    st.latex(r"\mathcal{L}_{L1}(G) = \mathbb{E}_{x,y}\big[\, \lVert y - G(x) \rVert_1 \,\big]")
    st.caption("x = degraded input, y = clean target, λ = 100 in this implementation.")

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("PSNR - Peak Signal-to-Noise Ratio")
    st.latex(r"\text{PSNR} = 10 \cdot \log_{10}\left(\frac{\text{MAX}_I^2}{\text{MSE}}\right)")
    st.latex(r"\text{MSE} = \frac{1}{mn}\sum_{i=0}^{m-1}\sum_{j=0}^{n-1}\big[I(i,j) - K(i,j)\big]^2")
    st.caption("Higher is better. Measures pixel-level fidelity between the enhanced/degraded image and the clean target.")

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("SSIM - Structural Similarity Index")
    st.latex(r"\text{SSIM}(x,y) = \frac{(2\mu_x\mu_y + c_1)(2\sigma_{xy} + c_2)}{(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}")
    st.caption("Range [-1, 1], 1 = identical. Captures luminance, contrast, and structural similarity rather than raw pixel error.")

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Crossing Number - classical minutiae extraction (used to pseudo-label YOLOv8)")
    st.latex(r"CN(p) = \frac{1}{2}\sum_{i=1}^{8} \big| \text{val}(p_i) - \text{val}(p_{i+1 \bmod 8}) \big|")
    st.markdown("""
    Computed on the 8-neighborhood of each skeletonized ridge pixel `p`:
    - **CN = 1** → ridge ending
    - **CN = 3** → bifurcation
    - **CN = 2** → normal ridge point (not a minutia)
    """)

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("Object detection metrics (YOLOv8)")
    st.latex(r"\text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN}")
    st.latex(r"\text{AP} = \int_0^1 p(r)\, dr \qquad \text{mAP} = \frac{1}{N}\sum_{i=1}^{N} AP_i")
    st.caption("mAP@50 uses IoU ≥ 0.5 to count a detection as correct; mAP@50-95 averages over IoU thresholds 0.5–0.95.")

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("FAR / FRR - biometric error rates")
    st.latex(r"\text{FAR}(\tau) = \frac{\#\{\text{impostor pairs with score} \geq \tau\}}{\#\{\text{impostor pairs}\}}")
    st.latex(r"\text{FRR}(\tau) = \frac{\#\{\text{genuine pairs with score} < \tau\}}{\#\{\text{genuine pairs}\}}")
    st.caption(
        f"At τ = {SOURCEAFIS['threshold']} (SourceAFIS's own documented ≈0.01% false-match reference point): "
        f"FAR = {FARFRR_AT_40['far']:.2f}%, FRR = {FARFRR_AT_40['frr']:.2f}% on this system."
    )

    st.markdown('<hr class="ridge">', unsafe_allow_html=True)
    docket("SourceAFIS match score")
    st.markdown("""
    SourceAFIS produces an open-ended similarity score (not a probability) from comparing two
    minutiae-based fingerprint templates - derived from the count and spatial/directional agreement
    of corresponding minutiae between probe and candidate. There is no universal fixed maximum;
    scores above ≈40 are SourceAFIS's own documented reference point for a confident genuine match.
    """)


# ============================================================
# FOOTER: PREV / NEXT PAGE NAVIGATION (shown under every page)
# ============================================================
st.markdown('<hr class="ridge">', unsafe_allow_html=True)


def _go_prev_page():
    idx = PAGES.index(st.session_state.nav_page)
    if idx > 0:
        st.session_state.nav_page = PAGES[idx - 1]


def _go_next_page():
    idx = PAGES.index(st.session_state.nav_page)
    if idx < len(PAGES) - 1:
        st.session_state.nav_page = PAGES[idx + 1]


_current_idx = PAGES.index(st.session_state.nav_page)
_nav_prev, _nav_mid, _nav_next = st.columns([1, 2, 1])

with _nav_prev:
    if _current_idx > 0:
        st.button("← Previous", use_container_width=True, on_click=_go_prev_page)

with _nav_mid:
    st.markdown(
        f"<div style='text-align:center; color:{MUTED}; font-family:JetBrains Mono, monospace; "
        f"font-size:0.75rem; padding-top:8px;'>Page {_current_idx + 1} of {len(PAGES)}</div>",
        unsafe_allow_html=True,
    )

with _nav_next:
    if _current_idx < len(PAGES) - 1:
        st.button("Next →", use_container_width=True, on_click=_go_next_page)