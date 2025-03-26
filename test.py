from Scripts.ML_Pipe.Recommend.test2 import *

if __name__ == '__main__':
    # main()
    # Or, to load a saved model and run a prediction example:
    df_raw = load_data("Derived_Data/feature/feature_20250301_105232.parquet")
    df_features = compute_features(df_raw)

    # Drop datetime columns if any remain
    drop_cols = [col for col in df_features.columns if pd.api.types.is_datetime64_any_dtype(df_features[col])]
    if drop_cols:
        logging.info(f"Dropping datetime columns: {drop_cols}")
        df_features = df_features.drop(columns=drop_cols)

    # Load the saved supervised model
    model_path = "models/recommend/model.pkl"
    saved_model = load(model_path)
    logging.info("Saved model loaded.")

    # Recreate the MDP from the computed features
    mdp = PitchSequencingMDP(df_features)
    mdp.estimate_transition_probabilities()
    mdp_policy = mdp.solve_mdp()

    # Prepare a sample game state for a given count, outs, inning, batter_id, and pitcher_id.
    game_state = prepare_game_state(
        count="1-1",
        outs=1,
        inning=6,
        batter_id=1000032366,
        pitcher_id=1000066910,
        hist_df=df_features
    )

    # Get a single recommended pitch using the loaded model.
    recommended_pitch = recommend_pitch(game_state, saved_model, mdp_policy, df_features)
    print("Recommended pitch:", recommended_pitch)

    # Simulate the next 10 pitches starting from the game state.
    pitch_sequence = simulate_next_pitches(game_state, saved_model, mdp_policy, mdp, df_features, n=10)
    print("Simulated pitch sequence:", pitch_sequence)