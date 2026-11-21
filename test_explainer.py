import main

df = main.add_phq_columns(main.load_data("data/DATASET-Raw_data_clean.csv"))
model = main.build_explainer_model()

samples = [df[df["level"] == level].head(2) for level in range(1, 6)]
samples.append(df[(df["level"] == 6) & (df["reasons"].apply(len) > 0)].head(3))
samples.append(df[(df["level"] == 6) & (df["reasons"].apply(len) == 0)].head(1))

for group in samples:
    for index, row in group.iterrows():
        level = int(row["level"])
        note = main.explain_reasons(level, row["reasons"], model)
        print("Row", index, "| level", level)
        print("Items:", row["reasons"])
        print("Note :", note)
        print()