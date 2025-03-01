import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Optimizing Pitch Sequence for Baseball",
    layout="wide"
)
  
data = pd.read_parquet("Derived_Data\clean\cleaned_20250228_194529.parquet")
data_clean = data.groupby(["PAofInning", "Inning", "Top/Bottom", 'PitcherId', "PitcherTeam"]).agg({
  'KorBB':lambda x: list(x),
  'PlayResult':lambda x: list(x),
  'PitchofPA':lambda x: list(x),
  'AutoPitchType':lambda x: list(x), ## Will combine with tag later
  'OutsOnPlay': "sum",
  "RunsScored": "sum"
}).reset_index()

data_clean["KorBB"] = data_clean["KorBB"].str.get(-1)

data_clean['PlayResult'] = data_clean['PlayResult'].str.get(-1)

data_clean['BatterResult'] = np.where(data_clean["KorBB"].isin(["Strikeout", "Walk"]), data_clean["KorBB"], data_clean["PlayResult"])

data_clean['LastPitch'] = data_clean['AutoPitchType'].str.get(-1)

data_clean["PitcherId"] = data_clean["PitcherId"].astype(dtype=str)

data_explode = data_clean.explode(["AutoPitchType", "PitchofPA"])


home, ml, outs, runs, appendix = st.tabs(["Home", "Machine Learning", "Out Analytics", "Run Analytics", "Appendix"])

with home:
    st.header("Pioneer Baseballs League Metrics Group Analysis")
    st.write(
        "Welcome to the Pioneer League Metrics Group interactive dashboard! Our mission is to leverage data visualization, machine learning models,\
        and statistical analysis to help teams within the Pioneer Baseball League (PBL) make informed, data-driven decisions. Our analysis focuses on \
        key baseball metrics to provide insights into player performance, team strategies, and game outcomes."
    )
    st.markdown("### Why This Matters")
    st.markdown("Data-driven approaches are transforming baseball. From sabermetrics to real-time game insights, teams that embrace analytics gain a competitive edge.\
                By applying machine learning models and statistical techniques, we can help optimize performance, enhance scouting, and refine in-game decision-making.")
    st.markdown("### Key Features:")
    st.markdown("- **Pitching Statistics**: Pitching and outs statics on the games")
    st.markdown("- **Team Performance**: Compare teams based on various metrics.")
    st.markdown("- **Strikeout Improvement**: Use machine learning to predict optimal pitches that should be thrown to get a player out.")
    st.markdown("- **Machine Learning**: For Strikeout Optimization Predicting the most effective pitches to retire batters.")
    st.markdown("- **Run Scoring Trends**: Identifying factors that influence offensive production.")


with ml:

    print("hello")


with outs:

    ## Filters for sidebar on teams, so only look at one teams pitchers at a time
    ## Also generalized model with bars for entire data

    print("hello")


with runs:

    st.title("Run Analytics")

    runs_per_pitch = data_explode.groupby(["AutoPitchType", "PitcherTeam"]).agg({"RunsScored":"mean"}).reset_index()

    colors = [
        "#B0B3BA",  # Light Cool Gray
        "#77B6B2",  # Brighter Teal
        "#E3CDA8",  # Warm Sand
        "#B5C99A",  # Olive Mist
        "#92B6D5",  # Dusty Blue
        "#D8A8A8",  # Rosy Blush
        "#C4A99D"   # Soft Mocha
    ]

    # Create a Plotly bar plot
    fig = px.bar(data, 
                x="AutoPitchType", 
                y="RunsScored", 
                color="AutoPitchType", 
                color_discrete_sequence=colors,
                facet_col="PitcherTeam", 
                title="Runs Scored by Pitch Type and Team",
                labels={"AutoPitchType": "Pitch Type", "RunsScored": "Runs Scored"},
                category_orders={"AutoPitchType": data["AutoPitchType"].unique()})

    # Customize the facet titles and make text darker
    fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1], font=dict(color='black')))

    # Remove the legend
    fig.update_layout(showlegend=False)
    
    # Show the plot in Streamlit
    st.title("MLB Pitch Analysis")
    st.plotly_chart(fig)  # Display Plotly chart in Streamlit       
            

with appendix:

    st.title("Appendix")

    # Convert data dictionary into an HTML table with bold column names
    data_dict_html = """
    <table>
        <thead>
            <tr>
                <th style="text-align: left;">Column Name</th>
                <th style="text-align: left;">Description</th>
            </tr>
        </thead>
        <tbody>
    """

    # Add each column and description with bold column names
    data_dict = [
        ("Date", "Game date (YYYY-MM-DD)."),
        ("Time", "Time of the pitch (HH:MM:SS)."),
        ("PAofInning", "Plate appearance number in the inning."),
        ("PitchofPA", "Pitch number within the plate appearance."),
        ("PitcherId", "Unique identifier for the pitcher."),
        ("BatterId", "Unique identifier for the batter."),
        ("PitcherThrows", "Pitcher's throwing hand (L/R)."),
        ("BatterSide", "Batter's hitting side (L/R)."),
        ("BatterTeam", "Team of the batter."),
        ("Inning", "Current inning of the game."),
        ("Top/Bottom", "Indicates if it's the top or bottom of the inning."),
        ("Outs", "Number of outs before the pitch."),
        ("Balls", "Count of balls before the pitch."),
        ("Strikes", "Count of strikes before the pitch."),
        ("PitchCall", "Outcome of the pitch (Ball, Strike, etc.)."),
        ("KorBB", "Strikeout (K) or walk (BB)."),
        ("CleanPitchType", "Categorized pitch type (e.g., Fastball, Slider)."),
        ("TaggedHitType", "Manually reviewed hit type (e.g., Line Drive, Fly Ball)."),
        ("PlayResult", "Result of the play (Single, Out, etc.)."),
        ("OutsOnPlay", "Number of outs recorded on the play."),
        ("RunsScored", "Number of runs scored on the play."),
        ("RelSpeed", "Pitch release speed (MPH)."),
        ("VertRelAngle", "Vertical release angle of pitch."),
        ("HorzRelAngle", "Horizontal release angle of pitch."),
        ("SpinRate", "Spin rate of the pitch (RPM)."),
        ("SpinAxis", "Spin axis of the pitch (degrees)."),
        ("Tilt", "Clock-face tilt of pitch spin."),
        ("RelHeight", "Pitch release height (feet)."),
        ("RelSide", "Lateral release position (feet)."),
        ("Extension", "Pitcher’s release extension (feet)."),
        ("VertBreak", "Total vertical movement of pitch (inches)."),
        ("InducedVertBreak", "Induced vertical movement of pitch (inches)."),
        ("HorzBreak", "Horizontal movement of pitch (inches)."),
        ("PlateLocHeight", "Vertical location of pitch at home plate (feet)."),
        ("PlateLocSide", "Horizontal location of pitch at home plate (feet)."),
        ("ZoneSpeed", "Speed of pitch when crossing the plate (MPH)."),
        ("VertApprAngle", "Vertical approach angle of pitch."),
        ("HorzApprAngle", "Horizontal approach angle of pitch."),
        ("ZoneTime", "Time for pitch to reach the plate (seconds)."),
        ("ExitSpeed", "Exit velocity of the ball off the bat (MPH)."),
        ("Angle", "Launch angle of the batted ball (degrees)."),
        ("Direction", "Horizontal spray angle of the ball."),
        ("HitSpinRate", "Spin rate of ball after contact (RPM)."),
        ("PositionAt110X", "X-coordinate of ball at 110 feet."),
        ("PositionAt110Y", "Y-coordinate of ball at 110 feet."),
        ("PositionAt110Z", "Z-coordinate of ball at 110 feet."),
        ("Distance", "Total distance traveled by the ball (feet)."),
        ("LastTrackedDistance", "Last recorded distance of ball (feet)."),
        ("Bearing", "Bearing angle of ball’s trajectory."),
        ("HangTime", "Time ball was in air before landing (seconds)."),
        ("pfxx", "Horizontal pitch movement."),
        ("pfxz", "Vertical pitch movement."),
        ("x0", "Initial X-position of the pitch."),
        ("y0", "Initial Y-position of the pitch."),
        ("z0", "Initial Z-position of the pitch."),
        ("vx0", "Initial X-velocity of pitch (ft/s)."),
        ("vy0", "Initial Y-velocity of pitch (ft/s)."),
        ("vz0", "Initial Z-velocity of pitch (ft/s)."),
        ("ax0", "Initial X-acceleration of pitch (ft/s²)."),
        ("ay0", "Initial Y-acceleration of pitch (ft/s²)."),
        ("az0", "Initial Z-acceleration of pitch (ft/s²)."),
        ("GameID", "Unique game identifier."),
        ("PitchUID", "Unique pitch identifier within the game."),
        ("EffectiveVelo", "Effective velocity of the pitch."),
        ("MaxHeight", "Maximum height of pitch (feet)."),
        ("MeasuredDuration", "Duration from pitch release to plate (seconds)."),
        ("SpeedDrop", "Speed reduction from release to plate (MPH)."),
        ("AutoHitType", "System-classified hit type (e.g., Ground Ball, Fly Ball)."),
        ("HitSpinAxis", "Spin axis of ball after contact (degrees)."),
    ]

    # Append rows to the HTML table
    for column_name, description in data_dict:
        data_dict_html += f"<tr><td><b>{column_name}</b></td><td>{description}</td></tr>"

    data_dict_html += "</tbody></table>"

    # Streamlit App
    st.markdown("## MLB Statcast Data Dictionary")
    st.write("This table describes all key columns in the dataset:")

    # Display the table using Markdown
    st.markdown(data_dict_html, unsafe_allow_html=True)




