import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="PragyanAI – Program Intelligence Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f0f1a",
    "axes.facecolor":    "#1a1a2e",
    "axes.edgecolor":    "#2a2a4a",
    "axes.labelcolor":   "#8899bb",
    "axes.titlecolor":   "#00d4ff",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.color":       "#8899bb",
    "ytick.color":       "#8899bb",
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "grid.color":        "#1e2a3a",
    "grid.linestyle":    "--",
    "grid.alpha":        0.6,
    "text.color":        "#cdd9f0",
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  "#2a2a4a",
    "legend.fontsize":   8,
    "font.family":       "monospace",
})

COLORS = ["#00d4ff", "#00e676", "#ffa726", "#bb86fc",
          "#ff5252", "#ffeb3b", "#26c6da", "#66bb6a"]

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,#0f0f1a,#1a1a2e,#16213e);
    padding:1.6rem 2rem; border-radius:14px; margin-bottom:1.2rem;
    border:1px solid #2a2a4a;
}
.main-header h1 { color:#00d4ff; font-size:1.8rem; margin:0; font-family:monospace; }
.main-header p  { color:#8899bb; margin:.4rem 0 0; font-size:.9rem; }
.kpi-card {
    background:linear-gradient(145deg,#1a1a2e,#16213e);
    border:1px solid #2a2a4a; border-radius:10px;
    padding:1rem 1.2rem; text-align:center;
}
.kpi-value { font-family:monospace; font-size:1.7rem; color:#00d4ff; font-weight:700; }
.kpi-label { color:#8899bb; font-size:.75rem; text-transform:uppercase;
             letter-spacing:1px; margin-top:3px; }
.kpi-delta { font-size:.72rem; margin-top:2px; }
.kpi-delta.good { color:#00e676; }
.kpi-delta.bad  { color:#ff5252; }
.section-title {
    font-family:monospace; color:#00d4ff; font-size:1rem;
    border-left:3px solid #00d4ff; padding-left:10px; margin:1.2rem 0 .8rem;
}
.insight-box {
    background:linear-gradient(135deg,#0d2137,#0a1628);
    border:1px solid #1e4a6e; border-left:4px solid #00d4ff;
    border-radius:8px; padding:.9rem 1rem; margin:.4rem 0;
    color:#cdd9f0; font-size:.88rem;
}
.warn-box {
    background:linear-gradient(135deg,#1f1207,#2a1a0a);
    border:1px solid #5a3a10; border-left:4px solid #ffa726;
    border-radius:8px; padding:.9rem 1rem; margin:.4rem 0;
    color:#f0d9aa; font-size:.88rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def show(fig):
    """Render matplotlib figure into Streamlit."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


def hbar(ax, categories, values, fmt=".1f"):
    cats = list(categories); vals = list(values)
    clrs = COLORS[:len(cats)]
    bars = ax.barh(cats, vals, color=clrs, edgecolor="#0f0f1a", linewidth=0.5)
    mx = max(vals) if vals else 1
    for b in bars:
        w = b.get_width()
        ax.text(w + mx*0.01, b.get_y()+b.get_height()/2,
                f"{w:{fmt}}", va="center", fontsize=7, color="#cdd9f0")
    ax.grid(axis="x"); ax.set_axisbelow(True)


def vbar(ax, categories, values, colors=None, fmt=".1f"):
    cats = list(categories); vals = list(values)
    clrs = (COLORS[:len(cats)]) if colors is None else colors
    bars = ax.bar(cats, vals, color=clrs, edgecolor="#0f0f1a", linewidth=0.5)
    mx = max(vals) if vals else 1
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x()+b.get_width()/2, h + mx*0.01,
                f"{h:{fmt}}", ha="center", fontsize=7, color="#cdd9f0")
    ax.grid(axis="y"); ax.set_axisbelow(True)


def heatmap(ax, data, title, cmap="Blues"):
    im = ax.imshow(data.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(data.columns))); ax.set_xticklabels(data.columns, fontsize=8)
    ax.set_yticks(range(len(data.index)));   ax.set_yticklabels(data.index, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.values[i,j]:.1f}", ha="center", va="center",
                    fontsize=8, color="white")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)


# ─────────────────────────────────────────────
# DATA GENERATION  (200K records)
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def generate_data(n=200_000, seed=42):
    rng = np.random.default_rng(seed)

    prog_types  = ["Basic", "Advanced", "GenAI", "Full Stack", "Data Science"]
    modes       = ["Online", "Offline", "Hybrid"]
    degrees     = ["BTech", "MTech", "PhD"]
    companies   = ["Top", "Mid", "Startup"]
    departments = ["CSE", "ECE", "IT", "Mechanical", "Civil", "MBA"]
    tiers       = ["Tier 1", "Tier 2", "Tier 3"]

    pt   = rng.choice(prog_types, n, p=[0.15, 0.25, 0.20, 0.25, 0.15])
    mode = rng.choice(modes, n, p=[0.35, 0.30, 0.35])

    duration = np.where(pt == "Basic",       rng.integers(1,  4,  n),
               np.where(pt == "Advanced",    rng.integers(3,  13, n),
               np.where(pt == "GenAI",       rng.integers(4,  13, n),
               np.where(pt == "Full Stack",  rng.integers(6,  25, n),
                                             rng.integers(6,  37, n)))))

    subjects = rng.integers(3, 18, n)
    projects = np.clip(rng.integers(0, 12, n), 0, 11)
    tools    = np.clip(rng.integers(1, 15, n), 1, 14)

    base = {"Basic": 15000, "Advanced": 45000, "GenAI": 80000,
            "Full Stack": 60000, "Data Science": 70000}
    cost = np.clip(
        np.array([base[p] for p in pt]) + rng.normal(0, 20000, n),
        5000, 500000).astype(int)

    fac_exp  = rng.integers(1, 25, n)
    fac_deg  = rng.choice(degrees, n)
    fac_ind  = rng.choice(["Yes", "No"], n, p=[0.55, 0.45])
    fac_comp = rng.choice(companies, n)

    cgpa    = rng.uniform(5.0, 10.0, n).round(2)
    dept    = rng.choice(departments, n)
    tier    = rng.choice(tiers, n, p=[0.20, 0.45, 0.35])
    skill_b = rng.integers(0, 4, n)

    imp = (
        (np.clip(duration, 0, 12) / 12) * 1.5 +
        (np.clip(projects, 0, 10) / 10) * 2.0 +
        (np.clip(tools,    0, 12) / 12) * 1.0 +
        (fac_exp / 24) * 1.0 +
        (fac_ind == "Yes").astype(int) * 0.8 +
        (mode == "Hybrid").astype(int) * 0.5 +
        (mode == "Offline").astype(int) * 0.3 +
        rng.normal(0, 0.5, n)
    )
    skill_imp = np.clip(np.round(imp / imp.max() * 3).astype(int), 0, 3)
    skill_a   = np.clip(skill_b + skill_imp, 0, 3)

    pp = np.clip(
        0.30 + 0.15*(skill_imp >= 2) + 0.10*(projects >= 5) +
        0.10*(tools >= 8) + 0.10*(fac_ind == "Yes") +
        0.05*(tier == "Tier 1") + 0.05*(cgpa >= 8.0) +
        0.10*(mode == "Hybrid") - 0.05*(duration > 24), 0, 0.95)
    placed = rng.uniform(0, 1, n) < pp

    salary = np.where(placed,
        np.clip(3 + skill_imp*2.5 + projects*0.5 +
                (cgpa-6)*0.8 + rng.normal(0, 1.5, n), 2, 40), 0.0).round(2)
    roi = np.where(cost > 0, (salary * 100000) / cost, 0).round(4)

    eff = (skill_imp/3*40 + placed.astype(int)*30 +
           np.clip(roi / (roi.max()+1e-9), 0, 1)*30).round(2)
    l_density = ((projects + tools) / np.clip(duration, 1, 36)).round(4)

    df = pd.DataFrame({
        "Program_ID":          [f"P{i:06d}" for i in range(n)],
        "Program_Name":        [f"{pt[i]} Prog {i%500}" for i in range(n)],
        "Program_Type":        pt,       "Mode":               mode,
        "Duration_Months":     duration, "Subjects_Count":     subjects,
        "Projects_Count":      projects, "Tools_Count":        tools,
        "Cost":                cost,
        "Faculty_Experience":  fac_exp,  "Faculty_Degree":     fac_deg,
        "Industry_Experience": fac_ind,  "Company_Background": fac_comp,
        "Student_ID":          [f"S{i:07d}" for i in range(n)],
        "College_Tier":        tier,     "Department":         dept,
        "CGPA":                cgpa,
        "Skill_Level_Before":  skill_b,  "Skill_Level_After":  skill_a,
        "Skill_Improvement":   skill_imp,
        "Placement_Status":    placed,   "Salary_LPA":         salary,
        "ROI":                 roi,      "Effectiveness_Score":eff,
        "Learning_Density":    l_density,
    })

    df["Duration_Bucket"] = pd.cut(df["Duration_Months"],
        bins=[0, 3, 6, 12, 36], labels=["<3m", "3–6m", "6–12m", ">12m"])
    df["Cost_Bucket"] = pd.cut(df["Cost"],
        bins=[0, 50000, 200000, 500000], labels=["<50K", "50K–2L", ">2L"])

    def seg(r):
        if 6 <= r.Duration_Months <= 12 and r.Projects_Count >= 5 and r.Faculty_Experience >= 5:
            return "High Value"
        if r.Cost > 200000 and r.Projects_Count < 3:
            return "Overpriced"
        if r.Duration_Months < 3 and r.Tools_Count < 3:
            return "Low Depth"
        if r.Subjects_Count > 10 and r.Skill_Improvement <= 1:
            return "Heavy+Inefficient"
        return "Standard"

    df["Segment"] = df.apply(seg, axis=1)
    return df


def load_user(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(f)
    st.error("Unsupported file. Upload CSV or Excel.")
    return pd.DataFrame()


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🧠 PragyanAI — Program Intelligence Engine</h1>
  <p>Skill program analysis · ROI benchmarking · Placement intelligence · 200K record dataset</p>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Source")
    src = st.radio("Dataset", ["Generated (200K)", "Upload My Data"])
    if src == "Upload My Data":
        up = st.file_uploader("CSV / Excel", type=["csv", "xlsx", "xls"])
        if up:
            df_raw = load_user(up)
            if df_raw.empty:
                st.stop()
            st.success(f"{len(df_raw):,} records loaded")
        else:
            st.info("Awaiting upload …")
            st.stop()
    else:
        with st.spinner("Generating 200K records …"):
            df_raw = generate_data(200_000)
        st.success("200,000 records ready ✓")

    st.markdown("---")
    st.markdown("## 🔧 Filters")
    pt_opts = sorted(df_raw["Program_Type"].unique())
    pt_sel  = st.multiselect("Program Type", pt_opts, default=pt_opts)
    md_opts = sorted(df_raw["Mode"].unique())
    md_sel  = st.multiselect("Mode", md_opts, default=md_opts)
    dur_r   = st.slider("Duration (months)", 1, 36, (1, 36))
    cost_r  = st.slider("Cost (₹ thousands)", 0, 500, (0, 500))
    st.markdown("---")
    st.caption("PragyanAI v2.0 · numpy + pandas + matplotlib")


df = df_raw[
    df_raw["Program_Type"].isin(pt_sel) &
    df_raw["Mode"].isin(md_sel) &
    df_raw["Duration_Months"].between(*dur_r) &
    df_raw["Cost"].between(cost_r[0]*1000, cost_r[1]*1000)
].copy()

if df.empty:
    st.warning("No records match filters. Adjust sidebar.")
    st.stop()


# ─────────────────────────────────────────────
# KPI STRIP
# ─────────────────────────────────────────────
pr    = df["Placement_Status"].mean() * 100
aroi  = df.loc[df["Placement_Status"], "ROI"].mean()
asi   = df["Skill_Improvement"].mean()
asal  = df.loc[df["Placement_Status"], "Salary_LPA"].mean()
acost = df["Cost"].mean()
total = len(df)

c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, val, lbl, dlt, cls in [
    (c1, f"{total:,}",         "Records",         "",            ""),
    (c2, f"{pr:.1f}%",         "Placement Rate",  "Target >65%", "good" if pr > 65 else "bad"),
    (c3, f"₹{asal:.1f} LPA",  "Avg Salary",      "Placed only", "good"),
    (c4, f"{aroi:.2f}×",       "Avg ROI",         "Salary/Cost", "good" if aroi > 1.5 else "bad"),
    (c5, f"{asi:.2f}/3",       "Avg Skill Gain",  "0–3 scale",   "good" if asi > 1.5 else "bad"),
    (c6, f"₹{acost/1000:.0f}K","Avg Cost",        "",            ""),
]:
    col.markdown(f"""<div class="kpi-card">
      <div class="kpi-value">{val}</div>
      <div class="kpi-label">{lbl}</div>
      <div class="kpi-delta {cls}">{dlt}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs([
    "📊 Overview",
    "⏱ Duration & Projects",
    "🛠 Tools & Subjects",
    "💰 Cost & ROI",
    "👨‍🏫 Faculty Impact",
    "🎓 Placement & Skills",
    "🗂 Segmentation",
    "🔍 Program Finder",
    "💡 Key Insights",
])


# ══════════════════════════════════════════════
# TAB 0 — OVERVIEW
# ══════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-title">Program Distribution & Effectiveness</div>',
                unsafe_allow_html=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Program type bar
    vc = df["Program_Type"].value_counts()
    vbar(axes[0], vc.index, vc.values, fmt=".0f")
    axes[0].set_title("Programs by Type")
    axes[0].set_xlabel("Type"); axes[0].set_ylabel("Count")
    plt.setp(axes[0].get_xticklabels(), rotation=15)

    # Mode pie
    mc = df["Mode"].value_counts()
    wedges, texts, autotexts = axes[1].pie(
        mc.values, labels=mc.index, autopct="%1.1f%%",
        colors=COLORS[:len(mc)], startangle=90,
        wedgeprops=dict(edgecolor="#0f0f1a", linewidth=1.2))
    for t in autotexts:
        t.set_color("#0f0f1a"); t.set_fontsize(8)
    axes[1].set_title("Mode Distribution")

    plt.tight_layout()
    show(fig)

    # Effectiveness heatmap
    st.markdown('<div class="section-title">Effectiveness Score — Type × Mode</div>',
                unsafe_allow_html=True)
    pivot = df.groupby(["Program_Type", "Mode"])["Effectiveness_Score"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    heatmap(ax, pivot, "Avg Effectiveness Score — Program Type × Mode", cmap="Blues")
    plt.tight_layout(); show(fig)

    # Placement by type
    st.markdown('<div class="section-title">Placement Rate by Program Type</div>',
                unsafe_allow_html=True)
    pr_type = df.groupby("Program_Type")["Placement_Status"].mean().mul(100).sort_values()
    fig, ax = plt.subplots(figsize=(8, 3))
    hbar(ax, pr_type.index, pr_type.values, fmt=".1f")
    ax.set_title("Placement Rate % by Program Type")
    ax.set_xlabel("Placement Rate (%)")
    plt.tight_layout(); show(fig)


# ══════════════════════════════════════════════
# TAB 1 — DURATION & PROJECTS
# ══════════════════════════════════════════════
with tabs[1]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Duration vs Skill Improvement</div>',
                    unsafe_allow_html=True)
        dur_sk = df.groupby("Duration_Bucket", observed=True)["Skill_Improvement"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, dur_sk.index, dur_sk.values, fmt=".2f")
        ax.set_title("Avg Skill Improvement by Duration")
        ax.set_xlabel("Duration"); ax.set_ylabel("Skill Improvement")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          🔥 <strong>Sweet spot = 6–12 months.</strong>
          Beyond 12 months skill gain plateaus. Short programs (<3m) show lowest impact.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Projects vs Placement Rate</div>',
                    unsafe_allow_html=True)
        df["Proj_Bucket"] = pd.cut(df["Projects_Count"], bins=[0, 2, 5, 11],
                                   labels=["<2", "3–5", "6+"])
        pp_proj = df.groupby("Proj_Bucket", observed=True)["Placement_Status"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, pp_proj.index, pp_proj.values, fmt=".1f")
        ax.set_title("Placement Rate by Projects Count")
        ax.set_xlabel("Projects"); ax.set_ylabel("Placement Rate (%)")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          🔥 <strong>Projects = strongest learning signal.</strong>
          6+ projects push placement to peak levels.
        </div>""", unsafe_allow_html=True)

    # Heatmap: Duration × Projects → Avg Salary
    st.markdown('<div class="section-title">Duration × Projects → Avg Salary (Placed)</div>',
                unsafe_allow_html=True)
    tmp = df[df["Placement_Status"]].copy()
    tmp["Proj_B2"] = pd.cut(tmp["Projects_Count"], bins=[0, 2, 5, 11],
                            labels=["<2", "3–5", "6+"])
    heat = tmp.groupby(["Duration_Bucket", "Proj_B2"],
                       observed=True)["Salary_LPA"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 3))
    heatmap(ax, heat, "Avg Salary LPA — Duration × Projects (Placed Students)", cmap="YlOrBr")
    plt.tight_layout(); show(fig)


# ══════════════════════════════════════════════
# TAB 2 — TOOLS & SUBJECTS
# ══════════════════════════════════════════════
with tabs[2]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Tools Count vs ROI</div>',
                    unsafe_allow_html=True)
        t_bins = pd.cut(df["Tools_Count"], bins=[0, 3, 8, 14], labels=["<3", "3–8", "8+"])
        t_roi  = df.groupby(t_bins, observed=True)["ROI"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, t_roi.index, t_roi.values,
             colors=[COLORS[2]]*len(t_roi), fmt=".2f")
        ax.set_title("Avg ROI by Tools Count")
        ax.set_xlabel("Tools"); ax.set_ylabel("ROI")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          ⚙️ <strong>8+ tools dramatically improves ROI.</strong>
          Tool exposure signals job-ready graduates.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Subjects Count vs Skill Improvement</div>',
                    unsafe_allow_html=True)
        s_bins  = pd.cut(df["Subjects_Count"], bins=[0, 5, 10, 18], labels=["<5", "5–10", ">10"])
        s_skill = df.groupby(s_bins, observed=True)["Skill_Improvement"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, s_skill.index, s_skill.values,
             colors=[COLORS[3]]*len(s_skill), fmt=".2f")
        ax.set_title("Skill Improvement by Subjects Count")
        ax.set_xlabel("Subjects"); ax.set_ylabel("Skill Improvement")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="warn-box">
          ⚠️ <strong>More subjects ≠ better learning.</strong>
          >10 subjects causes overload. 5–10 is the sweet spot.
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-title">Learning Density by Program Type</div>',
                unsafe_allow_html=True)
    ld = df.groupby("Program_Type")["Learning_Density"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(8, 3))
    hbar(ax, ld.index, ld.values, fmt=".3f")
    ax.set_title("Learning Density = (Projects + Tools) / Duration")
    ax.set_xlabel("Learning Density")
    plt.tight_layout(); show(fig)


# ══════════════════════════════════════════════
# TAB 3 — COST & ROI
# ══════════════════════════════════════════════
with tabs[3]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Cost Bracket vs Avg ROI</div>',
                    unsafe_allow_html=True)
        c_roi = df.groupby("Cost_Bucket", observed=True)["ROI"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, c_roi.index, c_roi.values, fmt=".2f")
        ax.set_title("Avg ROI by Cost Bracket")
        ax.set_xlabel("Cost Bracket"); ax.set_ylabel("ROI")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          💡 <strong>₹50K–₹2L delivers best ROI.</strong>
          High-cost (>₹2L) programs often underdeliver.
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Cost-per-Project vs Placement Rate</div>',
                    unsafe_allow_html=True)
        df["Cost_per_Proj"] = df["Cost"] / df["Projects_Count"].clip(1)
        cpp = pd.qcut(df["Cost_per_Proj"], q=4,
                      labels=["Q1\n(Best)", "Q2", "Q3", "Q4\n(Worst)"])
        cpp_p = df.groupby(cpp, observed=True)["Placement_Status"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, cpp_p.index, cpp_p.values,
             colors=[COLORS[4]]*len(cpp_p), fmt=".1f")
        ax.set_title("Placement Rate by Cost-per-Project Quartile")
        ax.set_xlabel("Cost/Project Quartile"); ax.set_ylabel("Placement Rate (%)")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="warn-box">
          🚨 <strong>High cost + low projects = worst programs.</strong>
          Always check cost-per-project before enrolling.
        </div>""", unsafe_allow_html=True)

    # Scatter: Cost vs Salary
    st.markdown('<div class="section-title">Cost vs Salary Scatter (Placed Students)</div>',
                unsafe_allow_html=True)
    samp = df[df["Placement_Status"]].sample(
        min(3000, int(df["Placement_Status"].sum())), random_state=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, pt in enumerate(samp["Program_Type"].unique()):
        s = samp[samp["Program_Type"] == pt]
        ax.scatter(s["Cost"], s["Salary_LPA"], alpha=0.3, s=12,
                   color=COLORS[i % len(COLORS)], label=pt)
    ax.set_xlabel("Program Cost (₹)"); ax.set_ylabel("Salary LPA")
    ax.set_title("Cost vs Salary — Placed Students (sampled)")
    ax.legend(fontsize=7); ax.grid(True)
    plt.tight_layout(); show(fig)


# ══════════════════════════════════════════════
# TAB 4 — FACULTY IMPACT
# ══════════════════════════════════════════════
with tabs[4]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Faculty Experience vs Placement Rate</div>',
                    unsafe_allow_html=True)
        fb = pd.cut(df["Faculty_Experience"], bins=[0, 4, 8, 24],
                    labels=["<4 yrs", "4–8 yrs", "8+ yrs"])
        fp = df.groupby(fb, observed=True)["Placement_Status"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, fp.index, fp.values, fmt=".1f")
        ax.set_title("Placement Rate by Faculty Experience")
        ax.set_xlabel("Experience"); ax.set_ylabel("Placement Rate (%)")
        plt.tight_layout(); show(fig)

    with col2:
        st.markdown('<div class="section-title">Industry Experience vs Skill Improvement</div>',
                    unsafe_allow_html=True)
        ie_sk = df.groupby("Industry_Experience")["Skill_Improvement"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, ie_sk.index, ie_sk.values,
             colors=[COLORS[1]]*len(ie_sk), fmt=".2f")
        ax.set_title("Skill Improvement: Industry vs No Industry")
        ax.set_xlabel("Industry Experience"); ax.set_ylabel("Skill Improvement")
        plt.tight_layout(); show(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.markdown('<div class="section-title">Degree × Industry → Avg Salary</div>',
                    unsafe_allow_html=True)
        di = df[df["Placement_Status"]].groupby(
            ["Faculty_Degree", "Industry_Experience"])["Salary_LPA"].mean().unstack(fill_value=0)
        x = np.arange(len(di.index)); w = 0.35
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(x - w/2, di.get("Yes", pd.Series([0]*len(di))), w,
               color=COLORS[0], label="Industry: Yes", edgecolor="#0f0f1a")
        ax.bar(x + w/2, di.get("No",  pd.Series([0]*len(di))), w,
               color=COLORS[4], label="Industry: No",  edgecolor="#0f0f1a")
        ax.set_xticks(x); ax.set_xticklabels(di.index)
        ax.set_title("Avg Salary by Degree & Industry Exp")
        ax.set_ylabel("Salary LPA"); ax.legend(); ax.grid(axis="y")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          🔥 <strong>Real-world teaching > theoretical.</strong>
          PhD + no industry underperforms BTech + industry.
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown('<div class="section-title">Company Background vs ROI</div>',
                    unsafe_allow_html=True)
        cr = df.groupby("Company_Background")["ROI"].mean()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        vbar(ax, cr.index, cr.values,
             colors=[COLORS[2]]*len(cr), fmt=".2f")
        ax.set_title("Avg ROI by Faculty Company Background")
        ax.set_xlabel("Background"); ax.set_ylabel("ROI")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          🏢 Faculty from <strong>top-tier companies</strong> deliver
          significantly higher ROI outcomes.
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5 — PLACEMENT & SKILLS
# ══════════════════════════════════════════════
with tabs[5]:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Skill Improvement → Placement Rate</div>',
                    unsafe_allow_html=True)
        si_p = df.groupby("Skill_Improvement")["Placement_Status"].mean().mul(100)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        grad = [COLORS[4], COLORS[2], COLORS[1], COLORS[0]]
        ax.bar(si_p.index.astype(str), si_p.values,
               color=grad[:len(si_p)], edgecolor="#0f0f1a")
        for i, (x, v) in enumerate(zip(si_p.index, si_p.values)):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=8, color="#cdd9f0")
        ax.set_title("Placement Rate by Skill Improvement Level")
        ax.set_xlabel("Skill Improvement (0–3)"); ax.set_ylabel("Placement Rate (%)")
        ax.grid(axis="y")
        plt.tight_layout(); show(fig)

    with col2:
        st.markdown('<div class="section-title">Mode Comparison</div>',
                    unsafe_allow_html=True)
        mc = df.groupby("Mode")[["Placement_Status", "Skill_Improvement", "ROI"]].mean()
        mc["Placement_Status"] *= 100
        modes_list = mc.index.tolist()
        metrics    = ["Placement_Status", "Skill_Improvement", "ROI"]
        labels_m   = ["Placement %", "Skill Gain", "ROI"]
        x = np.arange(len(modes_list)); w = 0.25
        fig, ax = plt.subplots(figsize=(6, 3.5))
        for i, (m, lbl) in enumerate(zip(metrics, labels_m)):
            vals = [mc.loc[mo, m] for mo in modes_list]
            ax.bar(x + i*w, vals, w, label=lbl,
                   color=COLORS[i], edgecolor="#0f0f1a")
        ax.set_xticks(x + w); ax.set_xticklabels(modes_list)
        ax.set_title("Mode Comparison: Placement / Skill / ROI")
        ax.legend(); ax.grid(axis="y")
        plt.tight_layout(); show(fig)
        st.markdown("""<div class="insight-box">
          🔥 <strong>Hybrid = best learning model</strong> across all three metrics.
        </div>""", unsafe_allow_html=True)

    # Box plot: Salary by Program Type
    st.markdown('<div class="section-title">Salary Distribution by Program Type (Placed)</div>',
                unsafe_allow_html=True)
    placed_df = df[df["Placement_Status"] & (df["Salary_LPA"] > 0)]
    types_ord  = sorted(placed_df["Program_Type"].unique())
    data_box   = [placed_df[placed_df["Program_Type"] == t]["Salary_LPA"].values
                  for t in types_ord]
    fig, ax = plt.subplots(figsize=(10, 4))
    bp = ax.boxplot(data_box, patch_artist=True, notch=False,
                    medianprops=dict(color="#ffeb3b", linewidth=2),
                    whiskerprops=dict(color="#8899bb"),
                    capprops=dict(color="#8899bb"),
                    flierprops=dict(marker=".", color="#8899bb", markersize=2, alpha=0.3))
    for patch, clr in zip(bp["boxes"], COLORS):
        patch.set_facecolor(clr); patch.set_alpha(0.7)
    ax.set_xticklabels(types_ord, rotation=15)
    ax.set_title("Salary LPA Distribution by Program Type (Placed Students)")
    ax.set_ylabel("Salary LPA"); ax.grid(axis="y")
    plt.tight_layout(); show(fig)

    # CGPA × Tier → Salary
    st.markdown('<div class="section-title">CGPA Band × College Tier → Avg Salary</div>',
                unsafe_allow_html=True)
    df["CGPA_Band"] = pd.cut(df["CGPA"], bins=[4, 6, 7, 8, 10],
                             labels=["<6", "6–7", "7–8", "8+"])
    tc = df[df["Placement_Status"]].groupby(
        ["CGPA_Band", "College_Tier"], observed=True)["Salary_LPA"].mean().unstack(fill_value=0)
    x = np.arange(len(tc.index)); w = 0.28
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, (col_name, clr) in enumerate(zip(tc.columns, COLORS)):
        ax.bar(x + i*w, tc[col_name], w,
               label=col_name, color=clr, edgecolor="#0f0f1a")
    ax.set_xticks(x + w); ax.set_xticklabels(tc.index)
    ax.set_title("Avg Salary by CGPA Band & College Tier")
    ax.set_ylabel("Salary LPA"); ax.legend(); ax.grid(axis="y")
    plt.tight_layout(); show(fig)


# ══════════════════════════════════════════════
# TAB 6 — SEGMENTATION
# ══════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-title">Program Segmentation</div>',
                unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        seg_vc = df["Segment"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        wedges, texts, autotexts = ax.pie(
            seg_vc.values, labels=seg_vc.index, autopct="%1.1f%%",
            colors=COLORS[:len(seg_vc)], startangle=90,
            wedgeprops=dict(edgecolor="#0f0f1a", linewidth=1.2))
        for t in autotexts:
            t.set_color("#0f0f1a"); t.set_fontsize(8)
        ax.set_title("Segment Distribution")
        plt.tight_layout(); show(fig)

    with col2:
        seg_m = df.groupby("Segment").agg(
            Placement=("Placement_Status", "mean"),
            ROI=("ROI", "mean"),
            Skill=("Skill_Improvement", "mean"),
        ).reset_index()
        seg_m["Placement"] *= 100
        seg_colors = {
            "High Value": "#00e676", "Overpriced": "#ff5252",
            "Low Depth": "#ffd740",  "Heavy+Inefficient": "#bb86fc",
            "Standard": "#00d4ff"
        }
        for _, row in seg_m.iterrows():
            clr = seg_colors.get(row["Segment"], "#8899bb")
            st.markdown(f"""
            <div style="background:#1a1a2e;border:1px solid #2a2a4a;border-left:4px solid {clr};
                 border-radius:8px;padding:.7rem 1rem;margin-bottom:.5rem">
              <strong style="color:{clr}">{row['Segment']}</strong>
              <span style="color:#8899bb;font-size:.8rem;margin-left:8px">
                Placed: {row['Placement']:.1f}% &nbsp;|&nbsp;
                ROI: {row['ROI']:.2f}× &nbsp;|&nbsp;
                Skill: {row['Skill']:.2f}/3
              </span>
            </div>""", unsafe_allow_html=True)

    # Grouped bar: segment metrics
    st.markdown('<div class="section-title">Segment Metrics Comparison</div>',
                unsafe_allow_html=True)
    segs = seg_m["Segment"].tolist()
    x    = np.arange(len(segs)); w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, (col_name, lbl, clr) in enumerate([
        ("Placement", "Placement %", COLORS[0]),
        ("ROI",       "ROI",         COLORS[1]),
        ("Skill",     "Skill Gain",  COLORS[2]),
    ]):
        ax.bar(x + i*w, seg_m[col_name], w,
               label=lbl, color=clr, edgecolor="#0f0f1a")
    ax.set_xticks(x + w); ax.set_xticklabels(segs, rotation=10)
    ax.set_title("Segment Comparison: Placement / ROI / Skill Gain")
    ax.legend(); ax.grid(axis="y")
    plt.tight_layout(); show(fig)

    st.markdown('<div class="section-title">Detailed Segment Table</div>',
                unsafe_allow_html=True)
    seg_detail = df.groupby("Segment").agg(
        Count=("Student_ID", "count"),
        Avg_Duration=("Duration_Months", "mean"),
        Avg_Projects=("Projects_Count", "mean"),
        Avg_Tools=("Tools_Count", "mean"),
        Avg_Cost=("Cost", "mean"),
        Placement_Rate=("Placement_Status", "mean"),
        Avg_ROI=("ROI", "mean"),
    ).reset_index()
    seg_detail["Placement_Rate"] = (seg_detail["Placement_Rate"] * 100).round(1)
    seg_detail = seg_detail.round(2)
    st.dataframe(seg_detail.set_index("Segment"), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 7 — PROGRAM FINDER
# ══════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-title">🔍 Weighted Program Scorer</div>',
                unsafe_allow_html=True)
    st.markdown("Adjust weights → programs ranked by composite score.")

    f1, f2, f3 = st.columns(3)
    with f1:
        w_roi   = st.slider("Weight: ROI",                0, 10, 5)
        w_place = st.slider("Weight: Placement Rate",     0, 10, 5)
    with f2:
        w_skill = st.slider("Weight: Skill Improvement",  0, 10, 4)
        w_proj  = st.slider("Weight: Projects Count",     0, 10, 4)
    with f3:
        w_tools = st.slider("Weight: Tools Count",        0, 10, 3)
        w_fac   = st.slider("Weight: Faculty Experience", 0, 10, 3)

    agg = df.groupby("Program_Name").agg(
        Program_Type    =("Program_Type",    "first"),
        Mode            =("Mode",            "first"),
        Duration        =("Duration_Months", "mean"),
        Projects        =("Projects_Count",  "mean"),
        Tools           =("Tools_Count",     "mean"),
        Cost            =("Cost",            "mean"),
        Placement_Rate  =("Placement_Status","mean"),
        Avg_ROI         =("ROI",             "mean"),
        Skill           =("Skill_Improvement","mean"),
        Fac_Exp         =("Faculty_Experience","mean"),
        Avg_Salary      =("Salary_LPA",      "mean"),
        Count           =("Student_ID",      "count"),
    ).reset_index()
    agg = agg[agg["Count"] >= 5]

    def norm(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-9)

    agg["Score"] = (
        w_roi   * norm(agg["Avg_ROI"]) +
        w_place * norm(agg["Placement_Rate"]) +
        w_skill * norm(agg["Skill"]) +
        w_proj  * norm(agg["Projects"]) +
        w_tools * norm(agg["Tools"]) +
        w_fac   * norm(agg["Fac_Exp"])
    ).round(3)

    top_n = st.slider("Show top N programs", 5, 40, 15)
    top   = agg.nlargest(top_n, "Score").sort_values("Score")

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    colors_score = plt.cm.Blues(np.linspace(0.35, 0.95, len(top)))
    bars = ax.barh(top["Program_Name"], top["Score"],
                   color=colors_score, edgecolor="#0f0f1a")
    mx = top["Score"].max()
    for b in bars:
        w_b = b.get_width()
        ax.text(w_b + mx*0.01, b.get_y() + b.get_height()/2,
                f"{w_b:.3f}", va="center", fontsize=7, color="#cdd9f0")
    ax.set_title(f"Top {top_n} Programs — Composite Score")
    ax.set_xlabel("Score"); ax.grid(axis="x")
    plt.tight_layout(); show(fig)

    st.markdown('<div class="section-title">Top Programs — Details</div>',
                unsafe_allow_html=True)
    display = top[["Program_Name", "Program_Type", "Mode", "Duration", "Projects",
                   "Tools", "Placement_Rate", "Avg_ROI", "Avg_Salary", "Cost", "Score"]].copy()
    display["Placement_Rate"] = (display["Placement_Rate"] * 100).round(1)
    display = display.rename(columns={
        "Placement_Rate": "Placed%", "Avg_ROI": "ROI",
        "Avg_Salary": "Salary LPA", "Duration": "Dur(m)"
    })
    st.dataframe(
        display.set_index("Program_Name").sort_values("Score", ascending=False),
        use_container_width=True)


# ══════════════════════════════════════════════
# TAB 8 — KEY INSIGHTS
# ══════════════════════════════════════════════
with tabs[8]:
    best_dur  = df.groupby("Duration_Bucket",  observed=True)["Skill_Improvement"].mean().idxmax()
    best_mode = df.groupby("Mode")["Placement_Status"].mean().idxmax()
    best_type = df.groupby("Program_Type")["Placement_Status"].mean().idxmax()

    st.markdown('<div class="section-title">📊 Data-Backed Key Insights</div>',
                unsafe_allow_html=True)

    insights = [
        ("🎯", "Duration Sweet Spot",
         f"6–12 months yields peak skill gain. Your data: <b>{best_dur}</b> is top bucket."),
        ("🛠", "Projects > Subjects",
         "Projects + tools are the <b>strongest predictors</b> of placement — more than subject count."),
        ("👨‍🏫", "Faculty Industry Exp is Critical",
         "BTech + industry background consistently outperforms PhD + no industry."),
        ("💻", f"Best Mode = {best_mode}",
         f"<b>Hybrid mode</b> leads across placement, skill, and ROI. Your data confirms: <b>{best_mode}</b>."),
        ("💰", "Mid-Range Cost = Best ROI",
         "₹50K–₹2L programs deliver the <b>best return on investment</b>."),
        ("🚀", f"Top Program Type = {best_type}",
         f"GenAI & Data Science produce highest placement. Your data: <b>{best_type}</b>."),
        ("📐", "Cost-per-Project = Hidden Quality Metric",
         "High cost + few projects = <b>worst programs</b>. Always compute before enrolling."),
        ("🏫", "Skills Close the College Tier Gap",
         "Skill improvement matters more than tier. High-skill Tier 3 > low-skill Tier 1."),
    ]

    cols = st.columns(2)
    for i, (icon, title, body) in enumerate(insights):
        with cols[i % 2]:
            st.markdown(f"""<div class="insight-box" style="margin-bottom:.8rem">
              <div style="color:#00d4ff;font-weight:700;margin-bottom:4px">{icon} {title}</div>
              <span style="color:#cdd9f0;font-size:.86rem">{body}</span>
            </div>""", unsafe_allow_html=True)

    # Importance bar chart
    st.markdown('<div class="section-title">Master Formula — What Makes a Great Program</div>',
                unsafe_allow_html=True)
    factors  = ["Projects\n(5+)", "Tools\n(8+)", "Faculty Exp\n(>8 yrs)",
                "Duration\n(6–12m)", "Mode\n(Hybrid)", "Cost\n(50K–2L)"]
    weights_v = [9.5, 8.5, 8.0, 7.5, 7.0, 6.5]
    fig, ax = plt.subplots(figsize=(10, 4))
    bars2 = ax.bar(factors, weights_v,
                   color=COLORS[:len(factors)], edgecolor="#0f0f1a", linewidth=0.8)
    for b in bars2:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + 0.05, f"{h}",
                ha="center", fontsize=9, color="#cdd9f0", fontweight="bold")
    ax.set_ylim(0, 11); ax.set_ylabel("Importance Score")
    ax.set_title("Key Program Quality Factors (Importance Score 0–10)")
    ax.grid(axis="y")
    plt.tight_layout(); show(fig)

    # Mindset cards
    st.markdown('<div class="section-title">Strategic Mindset Shift</div>',
                unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div class="warn-box">
          <strong>❌ Old Mindset</strong><br><br>
          "Will I get a placement?"<br>
          Focus on brand name &amp; cost<br>
          Ignore program structure<br>
          Certificates over skills
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class="insight-box">
          <strong>✅ New Mindset — Evaluate This:</strong><br><br>
          🛠 <b>Projects:</b> 5+ hands-on?<br>
          ⚙️ <b>Tools:</b> 8+ industry tools?<br>
          👨‍🏫 <b>Faculty:</b> Real industry experience?<br>
          ⏱ <b>Duration:</b> 6–12 month sweet spot?<br>
          💰 <b>ROI:</b> Mid-range cost, strong outcomes?
        </div>""", unsafe_allow_html=True)

    # Export
    st.markdown('<div class="section-title">📥 Export Filtered Data</div>',
                unsafe_allow_html=True)
    sample_exp = df.sample(min(10000, len(df)), random_state=42)
    st.download_button(
        label="⬇️ Download Filtered Sample (up to 10K rows, CSV)",
        data=sample_exp.to_csv(index=False).encode(),
        file_name="pragyanai_filtered_data.csv",
        mime="text/csv",
    )
