import pandas as pd
from pathlib import Path


RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "fatwas.parquet"


def normalize_fatwa_table(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """
    يحوّل جدول واحد من شكل Kaggle:
    questions, answers, titles, links, categories
    إلى أعمدة موحدة:
    id, question, answer, title, link, categories
    """

    cols = {c.lower(): c for c in df.columns}

    def pick(*names):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    q_col = pick("question", "questions")
    a_col = pick("answer", "answers")
    t_col = pick("title", "titles")
    l_col = pick("link", "links", "url")
    c_col = pick("categories", "category")

    if q_col is None or a_col is None:
        # هذا الملف ما يهمنا
        print(f"[SKIP] {path} (لا يحتوي على question(s)/answer(s))")
        return pd.DataFrame()

    rename_map = {
        q_col: "question",
        a_col: "answer",
    }
    if t_col:
        rename_map[t_col] = "title"
    if l_col:
        rename_map[l_col] = "link"
    if c_col:
        rename_map[c_col] = "categories"

    df = df.rename(columns=rename_map)

    # Add columns if missing
    for col in ["title", "link", "categories"]:
        if col not in df.columns:
            df[col] = None

    # Cleaning
    for col in ["question", "answer", "title"]:
        df[col] = df[col].astype(str).str.strip()

    # Remove empty rows
    df = df.dropna(subset=["question", "answer"])
    df = df[df["question"].str.len() > 0]
    df = df[df["answer"].str.len() > 0]

    return df[["question", "answer", "title", "link", "categories"]]


def load_fatwa_tables() -> pd.DataFrame:
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DIR}."
        )

    tables = []
    for path in RAW_DIR.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in {".csv", ".json"}:
            continue

        try:
            if suffix == ".csv":
                df = pd.read_csv(path)
            else:
                df = pd.read_json(path)
        except Exception as e:
            print(f"[SKIP] {path} (error while reading: {e})")
            continue

        norm = normalize_fatwa_table(df, path)
        if len(norm) == 0:
            continue

        print(f"[OK] Loaded {len(norm)} rows from {path}")
        tables.append(norm)

    if not tables:
        raise RuntimeError(
            "Not fit data/raw.\n"
            "Make sure the Kaggle file contains the columns: questions, answers, titles, links, categories."
        )

    df_all = pd.concat(tables, ignore_index=True)

    # Remove duplicates
    df_all = df_all.drop_duplicates(subset=["question", "answer"])

    # Add sequential id
    df_all.insert(0, "id", range(1, len(df_all) + 1))

    return df_all


def main():
    df = load_fatwa_tables()
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df)} fatwas to {OUT_PATH}")


if __name__ == "__main__":
    main()
