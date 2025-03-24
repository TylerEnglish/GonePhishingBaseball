from Scripts.ML_Pipe.ml_director import *

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Instead of training, we assume the models are already trained and stored.
    grab_load()  # This should log "All model files found. Ready to predict."

    '''
    Full intergrated
    '''
    # Run the combined prediction pipeline.
    df = predict(
        pitcher_id=1000066910.0,
        batter_id=1000032366.0,
        count="2-1",  # Balls - Strikes
        outs=2,
        inning=7,
        override_n_pitches=None  # Use regression result
    )
    print("\n\n\n\n=== Combined Predict Results ===") # -> outputs a dictionary of [{'pitch':{'ball':0.1231, 'strike':...,...}}]
    for idx, row in df.iterrows():
        print(f"Pitch {row['PitchNumber']} ({row['Pitch']}):")
        if row['Probabilities'] is not None:
            for cls, prob in row['Probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print("  No probability data.")
    print("\n=== End Combined Predict Results ===\n\n\n\n\n")

    '''
    Leave out regression
    '''
    df = predict(
        pitcher_id=1000066910.0,
        batter_id=1000032366.0,
        count="2-1",
        outs=2,
        inning=7,
        override_n_pitches=10  # Doesnt use Regression Result (can use a SD instead)
    )
    print("\n\n\n\n=== Combined Predict Results (no regression)===")
    for idx, row in df.iterrows():
        print(f"Pitch {row['PitchNumber']} ({row['Pitch']}):")
        if row['Probabilities'] is not None:
            for cls, prob in row['Probabilities'].items():
                print(f"  {cls}: {prob:.4f}")
        else:
            print("  No probability data.")
    print("\n=== End Combined Predict Results ===\n\n\n\n\n")

    '''
    Only Recommendation sequence
    '''
    # Recommend-only sequence.
    seq = predict_sequence(
        pitcher_id=1000066910.0,
        batter_id=1000032366.0,
        count="3-2",
        outs=2,
        inning=8,
        n=10
    )
    print(f"\n=== MDP-only Sequence: {seq} ===\n")
