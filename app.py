import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

st.set_page_config(
    page_title="Pioneer League Metrics Group",
    layout="wide",
    initial_sidebar_state="expanded",
)

colors = [
            "#5F7082",  # Muted denim blue
            "#4C8C65",  # Muted grass green
            "#B4976B",  # Soft tan (field dirt)
            "#7A5D42",  # Earthy brown (infield)
            "#5A7D8B",  # Muted stadium blue
            "#A35742",  # Duller red (team color)
            "#704B38",  # Leather brown (glove or ball)
            "#6B7C4B",  # Olive green (dugouts)
            "#D1B692",  # Light sandy beige (base dirt)
            "#8E7F5C",  # Muted golden brown (worn equipment)
            "#9C5D4F"   # Muted brick red (stadium walls)
        ]

# Sidebar Navigation
st.sidebar.header("Filters")

data = pd.read_parquet("Derived_Data/filter/filtered_20250228_231053.parquet")

team_names = {
    'GRE_VOY': 'Great Falls Voyagers',
    'BIL_MUS': 'Billings Mustangs',
    'GLA_RAN': 'Grand Junction Rockies',
    'BOI_HAW': 'Boise Hawks',
    'MIS_PAD': 'Missoula PaddleHeads',
    'OGD_RAP': 'Ogden Raptors',
    'NOR_COL1': 'Northern Colorado Owls',
    'IDA_CHU': 'Idaho Falls Chukars',
    'OAK_BAL': 'Oakland Ballers',
    'GLA_RAN1': 'Grand Junction Rockies',
    'YOL_HIG': 'Yolo High Wheelers',
    'ROC_VIB': 'Rocky Mountain Vibes',
    'GRA_ROC': 'Grand Rapids Rockies'
}

data["full_name"] = data["PitcherTeam"].map(team_names)
excluded_pitches = ["Knuckleball", "OneSeamFastBall", "Sweeper", "Other", "None"]
data = data.loc[~data["CleanPitchType"].isin(excluded_pitches), ]

pitcher_team = list(data["full_name"].unique()) + ["All Teams"]
pitcher_team.sort()
# Dropdown menu with default set to 'IDA_CHU'
selected_pitcher_team = st.sidebar.selectbox(
    "Select a Pitcher's Team",
    options=pitcher_team,
    index=list(pitcher_team).index("Idaho Falls Chukars"),  # Set default to 'IDA_CHU'
)

if selected_pitcher_team != "All Teams":
    data = data[data["full_name"] == selected_pitcher_team]


home, ml, outs, runs, appendix = st.tabs(
    ["Home", "Machine Learning", "Out Analytics", "Run Analytics", "Appendix"]
)

with home:
    st.header("Pioneer Baseballs League Metrics Group Analysis")
    st.write(
        "Welcome to the Pioneer League Metrics Group interactive dashboard! Our mission is to leverage data visualization, machine learning models,\
        and statistical analysis to help teams within the Pioneer Baseball League (PBL) make informed, data-driven decisions. Our analysis focuses on \
        key baseball metrics to provide insights into player performance, team strategies, and game outcomes."
    )
    st.markdown("### Why This Matters")
    st.markdown(
        "Data-driven approaches are transforming baseball. From sabermetrics to real-time game insights, teams that embrace analytics gain a competitive edge.\
                By applying machine learning models and statistical techniques, we can help optimize performance, enhance scouting, and refine in-game decision-making."
    )
    st.markdown("### Key Features:")
    st.markdown("- **Pitching Statistics**: Pitching and outs statics on the games")
    st.markdown("- **Team Performance**: Compare teams based on various metrics.")
    st.markdown(
        "- **Strikeout Improvement**: Use machine learning to predict optimal pitches that should be thrown to get a player out."
    )
    st.markdown(
        "- **Machine Learning**: For Strikeout Optimization Predicting the most effective pitches to retire batters."
    )
    st.markdown(
        "- **Run Scoring Trends**: Identifying factors that influence offensive production."
    )


with ml:
    # File uploader
    st.title("Machine Learning")

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx","parquet"])
    if uploaded_file is not None:
        # Check file type and load accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)


with outs:
    st.title("Out Analytics")
    
    # Count the number of occurrences of each pitch type
    pitch_counts = data["CleanPitchType"].value_counts().reset_index()
    pitch_counts.columns = ["CleanPitchType", "Count"]

    pitch_counts = pitch_counts.sort_values('Count', ascending=False)

    pitch_types = pitch_counts['CleanPitchType'].unique()

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        with st.container(border=True):
            with col3:
                top_pitches = st.number_input("Top N Pitch Counts", min_value=3, max_value = len(pitch_types), value=5, step=1, format="%d")

        top_n_pitches = pitch_types[:top_pitches]
        pitch_counts = pitch_counts[pitch_counts["CleanPitchType"].isin(top_n_pitches)]

        # Create a bar chart
        fig = px.bar(pitch_counts, x="CleanPitchType", y="Count",
                    labels={"CleanPitchType": "Pitch Type", "Count": "Number of Throws"},
                    color="CleanPitchType",
                    color_discrete_sequence=colors)
        fig.update_layout(showlegend=False)
        # Show the plot
        st.markdown(f"## **Count of Pitch Types for {selected_pitcher_team}**")
        st.plotly_chart(fig)

    
    with st.container(border=True):
        total_pitches = data["CleanPitchType"].value_counts().reset_index()
        total_pitches.columns = ["CleanPitchType", "TotalPitches"]
        # Count pitches that resulted in a strike (called or swinging)
        strike_pitches = data[data["PitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
        strike_counts = strike_pitches["CleanPitchType"].value_counts().reset_index()
        strike_counts.columns = ["CleanPitchType", "StrikePitches"]
        # Merge both counts
        pitch_strike_rates = total_pitches.merge(strike_counts, on="CleanPitchType", how="left").fillna(0)
        # Calculate the percentage of throws that resulted in a strike
        pitch_strike_rates["StrikePercentage"] = (pitch_strike_rates["StrikePitches"] / pitch_strike_rates["TotalPitches"])


        pitch_strike_rates = pitch_strike_rates.sort_values('StrikePercentage', ascending=False)

        pitch_types = pitch_strike_rates['CleanPitchType'].unique()
        col1, col2, col3 = st.columns(3)
        with st.container(border=True):
            with col3:
                top_pitches = st.number_input("Top N Pitches Strikes", min_value=3, max_value = len(pitch_types), value=5, step=1, format="%d")

        top_n_pitches = pitch_types[:top_pitches]
        pitch_strike_rates = pitch_strike_rates[pitch_strike_rates["CleanPitchType"].isin(top_n_pitches)]

        # Bar chart of percentage of throws resulting in strike
        fig = px.bar(pitch_strike_rates, x="CleanPitchType", y="StrikePercentage",
                    labels={"CleanPitchType": "Pitch Type", "StrikePercentage": "Strike Percentage"},
                    color="CleanPitchType",
                    color_discrete_sequence=colors)
        # Remove legend
        fig.update_layout(showlegend=False)
        fig.update_yaxes(tickformat=".0%")
        # Show the plot
        st.markdown(f"## **Percentage of Throws Resulting in Strike for {selected_pitcher_team}**")
        st.plotly_chart(fig)

with runs:
    data_clean = (
        data.groupby(["PAofInning", "Inning", "Top/Bottom", "full_name"])
        .agg(
            {
                "KorBB": lambda x: list(x),
                "PlayResult": lambda x: list(x),
                "PitchofPA": lambda x: list(x),
                "CleanPitchType": lambda x: list(x),  ## Will combine with tag later
                "OutsOnPlay": "sum",
                "RunsScored": "sum",
            }
        )
        .reset_index()
    )

    data_clean["KorBB"] = data_clean["KorBB"].str.get(-1)

    data_clean["PlayResult"] = data_clean["PlayResult"].str.get(-1)

    data_clean["BatterResult"] = np.where(
        data_clean["KorBB"].isin(["Strikeout", "Walk"]),
        data_clean["KorBB"],
        data_clean["PlayResult"],
    )

    data_clean["LastPitch"] = data_clean["CleanPitchType"].str.get(-1)

    data_explode = data_clean.explode(["CleanPitchType", "PitchofPA"])

    st.title("Run Analytics")

    with st.container(border=True):
        runs_per_pitch = (
            data_clean.loc[data_clean["RunsScored"] != 0].groupby(["LastPitch"])
                                 .agg(count = ("RunsScored", "sum")).reset_index()
        )

        runs_per_pitch = runs_per_pitch.sort_values('count')

        pitch_types = runs_per_pitch['LastPitch'].unique()
        col1, col2, col3 = st.columns(3)
        with st.container(border=True):
            with col3:
                top_pitches = st.number_input("Bottom N Pitches for Runs Scored", min_value=3, max_value=len(pitch_types), value=5, step=1, format="%d")

        top_n_pitches = pitch_types[:top_pitches]
        runs_per_pitch = runs_per_pitch[runs_per_pitch["LastPitch"].isin(top_n_pitches)]

        # Create a Plotly bar plot
        fig = px.bar(
            runs_per_pitch,
            x="LastPitch",
            y="count",
            color="LastPitch",
            barmode="stack",
            color_discrete_sequence=colors,
            labels={"LastPitch": "Pitch Type", "count": "Runs Scored"},
        )

        # Customize the facet titles and make text darker
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1], font=dict(color="black"))
        )

        # Remove the legend
        fig.update_layout(showlegend=False)

        # Show the plot in Streamlit
        st.markdown(f"## **Runs by Pitch Type for {selected_pitcher_team}**")
        st.plotly_chart(fig)  # Display Plotly chart in Streamlit
    
    with st.container(border=True):
        # Last pitch with Play Result
        two_results = data_explode
        two_results =  two_results[two_results["BatterResult"].isin(["Single", "Double", "Triple", "HomeRun", "Walk", "Sacrifice", "Out", "Strikeout"])]

        two_results["BatterResult"] = np.where(two_results["BatterResult"].isin(["Out", "Strikeout"]), "Out", "On Base")

        last_pitch_result = (
            two_results.groupby(["CleanPitchType", "BatterResult"])
            .agg(count = ("RunsScored", "count"))
            .reset_index()
        )

        # bottom_pitches
        outs = last_pitch_result[last_pitch_result["BatterResult"].isin(["Out", "Strikeout"])]
        outs = outs.sort_values('count')

        pitch_types = outs['CleanPitchType'].unique()

        col1, col2, col3 = st.columns(3)
        with st.container(border=True):
            with col3:
                top_pitches = st.number_input("Bottom N Pitches for On Base", min_value=3, max_value=len(pitch_types), value=5, step=1, format="%d")

        top_n_pitches = pitch_types[:top_pitches]
        last_pitch_result = last_pitch_result[last_pitch_result["CleanPitchType"].isin(top_n_pitches)]
        sorted_result = pd.concat([
            last_pitch_result[last_pitch_result['BatterResult'].isin(["Out", "Strikeout"])].sort_values(by='count', ascending=False),
            last_pitch_result[~last_pitch_result['BatterResult'].isin(["Out", "Strikeout"])]
        ])
        # Create a Plotly bar plot
        fig = px.bar(
            sorted_result,
            x="CleanPitchType",
            y="count",
            color="BatterResult",
            barmode='group',
            color_discrete_sequence=colors,
            labels={"CleanPitchType": "Pitch Type", "count": "Result of At Bat Count"},
        )

        # Customize the facet titles and make text darker
        fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1], font=dict(color="black"))
        )

        # Show the plot in Streamlit
        st.markdown(f"## **Outs VS. On Base for {selected_pitcher_team}**")
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
        ("CleanPitchType", "Categorized pitch type (e.g., Fastball, Slider) and combines TaggedPitchType and AutoPitchType."),
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
        data_dict_html += (
            f"<tr><td><b>{column_name}</b></td><td>{description}</td></tr>"
        )

    data_dict_html += "</tbody></table>"

    # Streamlit App
    st.markdown("## MLB Statcast Data Dictionary")
    st.write("This table describes all key columns in the dataset:")

    # Display the table using Markdown
    st.markdown(data_dict_html, unsafe_allow_html=True)
