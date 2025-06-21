def create_sequences(claims_df):
    """
    Creates sequences of diagnoses for each patient as strings.
    
    Args:
        claims_df: DataFrame containing claims data
        
    Returns:
        List of strings, where each string is a patient's diagnoses joined by '|',
        maintaining chronological order
    """
    # Sort claims by date
    claims_df = claims_df.sort_values(['patient', 'date'])
    
    # Group by patient and create sequences
    sequences = []
    
    for _, patient_claims in claims_df.groupby('patient'):
        # Get diagnosis codes and handle NaN values
        diagnoses = patient_claims['dx'].dropna().tolist()
        
        # Only include sequences that meet minimum length
        if len(diagnoses) >= 2:  # Need at least 2 codes for input-target pairs
            # Join the diagnoses into a single string with '|' separator
            sequence_str = ' '.join(diagnoses)
            sequences.append(sequence_str)
    
    print(f"Created {len(sequences)} patient sequences")
    return sequences