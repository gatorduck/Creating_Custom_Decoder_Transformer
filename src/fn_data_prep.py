
def create_sequences(df):
    # 1. Sort by patient and date
    df_sorted = df.sort_values(['patient', 'date'])
    # 2. Group by patient
    grouped = df_sorted.groupby('patient')
    # 3. Aggregate dx codes into a space-separated string for each patient
    sequences = grouped['dx'].apply(lambda codes: ' '.join(str(code) for code in codes))
    return sequences