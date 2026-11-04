"""
[app](https://capstone-data-science-project-dktwfa44tpcrsecbtmoptq.streamlit.app/)
Treatment recommender - user-based collaborative filtering (cosine similarity)
Target: Treatment
Run:  pip install streamlit pandas numpy
      streamlit run app.py
"""
import re
import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Treatment Recommender", page_icon="🩺", layout="wide")

# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading and cleaning data...")
def load(raw: bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw))
    # same cleaning as the notebook
    df = df.drop(columns=[c for c in df.columns
                          if c.lower() == "timestamp" or re.search(r"\.\d+$", c)])
    if "SelfEmployed" in df.columns:
        df["SelfEmployed"] = df["SelfEmployed"].bfill()
    df = df.drop_duplicates()

    target = next((c for c in df.columns
                   if re.sub(r"[^a-z]", "", c.lower()) == "treatment"), None)
    if target is None:
        raise ValueError("No 'Treatment' column found in the CSV.")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    feats = [c for c in df.columns if c != target]
    df[feats] = df[feats].fillna("Unknown").astype(str)
    df[target] = df[target].astype(str)

    X = np.empty((len(df), len(feats)), dtype=np.int16)
    labels = {}
    for j, c in enumerate(feats):
        codes, uniques = pd.factorize(df[c], sort=True)
        X[:, j] = codes
        labels[c] = list(uniques)
    y, classes = pd.factorize(df[target], sort=True)
    return df, feats, target, X, y.astype(np.int64), labels, list(classes)


# ------------------------------------------------- collaborative filtering
# One-hot cosine similarity between two patients = (shared answers) / F,
# so we only need to count how many answers each patient shares with the query.
def class_hist(q, X, y, n_classes, F):
    shared = (X == q).sum(axis=1)
    return np.bincount(shared * n_classes + y,
                       minlength=(F + 1) * n_classes).reshape(F + 1, n_classes)


def vote(h, k, F):
    """Use the k most similar patients (ties at the cut-off split
    proportionally) and return similarity-weighted class scores."""
    n_classes = h.shape[1]
    scores, rem, buckets = np.zeros(n_classes), k, []
    for m in range(F, 0, -1):
        tot = h[m].sum()
        if tot == 0:
            continue
        take = min(rem, tot)
        sim = m / F
        scores += sim * h[m] * (take / tot)
        buckets.append((m, sim, take, h[m] / tot))
        rem -= take
        if rem <= 0:
            break
    s = scores.sum()
    return (scores / s if s else scores), buckets


def evaluate(X, y, n_classes, F, grid, n_test=500, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    test_idx = idx[: int(len(y) * 0.2)]
    train_idx = idx[int(len(y) * 0.2):]
    Xtr, ytr = X[train_idx], y[train_idx]
    sample = test_idx[:n_test]
    correct = np.zeros(len(grid))
    for p in sample:
        h = class_hist(X[p], Xtr, ytr, n_classes, F)
        for gi, k in enumerate(grid):
            sc, _ = vote(h, k, F)
            correct[gi] += int(np.argmax(sc) == y[p])
    baseline = (y[sample] == np.bincount(ytr).argmax()).mean()
    return correct / len(sample), baseline, len(sample)


# ------------------------------------------------------------------ UI
st.title("🩺 Mental-health treatment recommender")
st.caption("User-based collaborative filtering with cosine similarity. Target: **Treatment**. "
           "Decision-support demo only, not a medical diagnosis.")

with st.sidebar:
    st.header("1 · Data")
    up = st.file_uploader("Upload Mental.csv", type="csv")
    if up is not None:
        raw = up.getvalue()
    elif os.path.exists("Mental.csv"):
        raw = open("Mental.csv", "rb").read()
        st.caption("Using Mental.csv from the app folder.")
    else:
        st.info("Upload Mental.csv to begin.")
        st.stop()

try:
    df, feats, target, X, y, labels, classes = load(raw)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()

F, C = len(feats), len(classes)

with st.sidebar:
    st.success(f"{len(df):,} patients · {F} features")
    st.header("2 · Neighbours")
    st.session_state.setdefault("k", 200)
    k = st.slider("k (most similar patients)", 5, 2000, key="k", step=5)

# --- profile form
def random_patient():
    i = np.random.randint(len(df))
    for j, c in enumerate(feats):
        st.session_state[f"sel_{c}"] = df.at[i, c]

st.subheader("Describe the individual")
st.button("🎲 Load random patient", on_click=random_patient)
cols = st.columns(3)
profile = {}
for j, c in enumerate(feats):
    profile[c] = cols[j % 3].selectbox(c, labels[c], key=f"sel_{c}")

q = np.array([labels[c].index(profile[c]) for c in feats], dtype=np.int16)

# --- recommendation (updates live whenever an input changes)
h = class_hist(q, X, y, C, F)
scores, buckets = vote(h, k, F)
best = int(np.argmax(scores))
is_yes = classes[best].lower() == "yes"

st.divider()
st.subheader("Recommendation")
m1, m2 = st.columns([1, 2])
m1.metric("Result", "Seek treatment" if is_yes else "No treatment indicated",
          f"Treatment = {classes[best]}", delta_color="off")
m1.metric("Confidence", f"{scores[best]:.1%}", f"from {k} most similar patients",
          delta_color="off")
if scores[best] < 0.60:
    m1.warning("Low confidence: similar patients are split. Treat as inconclusive.")
m2.bar_chart(pd.DataFrame({"Similarity-weighted vote": scores}, index=classes))

st.markdown("**Closest matching patient groups**")
tbl = pd.DataFrame(
    [{"Shared answers": f"{m} of {F}", "Cosine": round(s, 3),
      "Patients used": int(n), **{f"% {cl}": f"{p[i]:.1%}" for i, cl in enumerate(classes)}}
     for m, s, n, p in buckets[:8]])
st.dataframe(tbl, hide_index=True, use_container_width=True)

# --- evaluation
st.divider()
st.subheader("Model check (hold-out)")
if st.button("Evaluate and find best k"):
    grid = [5, 15, 25, 50, 100, 200, 400, 800, 1500]
    with st.spinner("Testing on 500 held-out patients..."):
        acc, base, n = evaluate(X, y, C, F, grid)
    res = pd.DataFrame({"Collaborative filtering": acc,
                        "Majority baseline": base}, index=grid)
    res.index.name = "k"
    st.line_chart(res)
    st.write(f"Best k = **{grid[int(acc.argmax())]}** with accuracy **{acc.max():.1%}** "
             f"versus a majority baseline of {base:.1%} (tested on {n} patients).")