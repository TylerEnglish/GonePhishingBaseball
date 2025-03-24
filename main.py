'''
Run Model
'''
from Scripts.ML_Pipe.Recommend.model import *

pretrained_model_path = "models/recommend/model.pkl"
zip_path = "models/recommend/model.zip"
with zipfile.ZipFile(zip_path, 'r') as zipf:
    zipf.extractall(os.path.dirname(pretrained_model_path))
logging.info("Model unzipped.")


df_raw = load_data("Derived_Data/feature/feature_20250301_105232.parquet")
df_features = compute_features(df_raw)
mdp = PitchSequencingMDP(df_features)
mdp.estimate_transition_probabilities()
mdp_policy = mdp.solve_mdp()
pretrained_model = load(pretrained_model_path)

game_state = prepare_game_state(
    count='1-1',
    outs=1,
    inning=6,
    batter_id=1000032366,
    pitcher_id=1000066910,
    hist_df=df_features
)

rec_pitch = recommend_pitch(game_state, pretrained_model, mdp_policy, df_features)
logging.info("Batter Recommended Pitch: %s", rec_pitch)

next_pitches = simulate_next_pitches(game_state, pretrained_model, mdp_policy, mdp, df_features, n=10)
logging.info("Simulated next pitches: %s", next_pitches)
print(f"\n\n\n=================================================================================================================================================================================\n\n\n")
print(f"\t\t\t{next_pitches}")
print(f"\n\n\n=================================================================================================================================================================================")
