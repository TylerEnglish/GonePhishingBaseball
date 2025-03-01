import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from Scripts.ML_Pipe.Base_Models.base_director import predict

st.set_page_config(
    page_title="Pioneer League Machine Learning Group",
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

data = pd.read_parquet("Derived_Data/filter/filtered_20250301_104903.parquet")

data["year"] = pd.to_datetime(data["Date"]).dt.year
data["year"] = data["year"].fillna(method="ffill")

years = list(data["year"].unique()) + ["All Years"]

selected_year = st.sidebar.selectbox(
    "Select a Year",
    options=years,
    index=list(years).index("All Years"), 
)

if selected_year != "All Years":
    data = data[data["year"] == selected_year]

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

data["TeamName"] = data["PitcherTeam"].map(team_names)
excluded_pitches = ["Knuckleball", "OneSeamFastBall", "Sweeper", "Other", "None"]
data = data.loc[~data["CleanPitchType"].isin(excluded_pitches), ]

pitcher_team = list(data["TeamName"].unique()) + ["All Teams"]
pitcher_team.sort()
# Dropdown menu with default set to 'IDA_CHU'
selected_pitcher_team = st.sidebar.selectbox(
    "Select a Pitcher's Team",
    options=pitcher_team,
    index=list(pitcher_team).index("Idaho Falls Chukars"),  # Set default to 'IDA_CHU'
)

if selected_pitcher_team != "All Teams":
    data = data[data["TeamName"] == selected_pitcher_team]

home, ml, outs, runs, outs_by, appendix, future = st.tabs(
    ["Home", "Machine Learning", "Out Analytics", "Run Analytics", "Strikes and Outs", "Appendix", "Future Goals"]
)

with home:
    st.header("Pioneer Baseball League: Optimized Pitching Sequence Model")
    st.write(
        "Welcome to the Pioneer League Machine Learning Group interactive dashboard! Our mission is to leverage data for machine learning models\
        and statistical analysis to help teams within the Pioneer Baseball League (PBL) make informed, data-driven decisions. Our project focuses on\
        using pitcher and batter data to create an optimized pitching sequence that predicts, for a given pitcher and batter, how many pitches it would take to get an out as well as\
        the types of pitches to use."
    )
    st.markdown("### Why This Matters")
    st.markdown(
        "Data-driven approaches are transforming baseball. From sabermetrics to real-time game insights, teams that embrace analytics gain a competitive edge.\
        By applying our nested machine learning model, we can help optimize performance for pitchers so that their team can get back in the batter's box and scoring runs."
    )
    st.markdown("### Machine Learning: Models and Validation Metrics")
    st.markdown("""
                * Models that we used include: **XGBoost**, **TabNet**, and **PyCaret**
                    * We originally planned on using TabNet for both the regression and categorization of our nested model;\
                      however that was not the case. We ended up running PyCaret to see which model best fit our data and chose XGBoost for the regression model.
                * For the classification model with TabNet we used **accuracy**, **F1**, and **precision**.
                    * We decided to use accuracy as a typical metric, along with F1 as a good metric for balance between precision and recall,\
                      and precision, specifically, since we are being most cautious with false positives for this model since we don't want to\
                      expect to have an out or strike and get a ball in play or worse a homerun. 
                """)

with ml:
    # File uploader
    st.title("Machine Learning")
    col1, col2, col3 = st.columns(3)

    
    with col2:
        st.markdown(
            """
            <div style="text-align: center; margin-left: 5%;">
                <h4>Best Metrics Achieved</h4>
            </div>
            """,
            unsafe_allow_html=True
        )
        score = pd.read_csv("Derived_Data/model_pred/scores.csv")
        score.columns = ["Accuracy", "F1", "Precision"]
        st.markdown(score.style.hide(axis="index").to_html(), unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx","parquet"])
    if uploaded_file is not None:
        # Check file type and load accordingly
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".parquet"):
            df = pd.read_parquet(uploaded_file)

        unique_pitchers = list(df["PitcherId"].unique())
        unique_batters = list(df["BatterId"].unique())

        pitcher_id = [1000066910.0,1000060505.0,701628.0,815136.0,1000056876.0]
        batter_id = [1000032366.0, 1000274194.0,1000035496.0,1000056633.0,683106.0]

        predictions = predict(pitcher_id[0], batter_id[0])
        
        cls_pred = f"Derived_Data/model_pred/cls_prediction_report_{int(pitcher_id[0])}_{int(batter_id[0])}.csv"
        reg_pred = f"Derived_Data/model_pred/reg_prediction_report_{int(pitcher_id[0])}_{int(batter_id[0])}.csv"

        reg = pd.read_csv(reg_pred)
        cls = pd.read_csv(cls_pred)

        st.table(reg)
        st.table(cls)


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
        strike_pitches = data[data["CleanPitchCall"].isin(["Strike"])]
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
        data.groupby(["PAofInning", "Inning", "Top/Bottom", "TeamName"])
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
        on_base = last_pitch_result[~last_pitch_result["BatterResult"].isin(["Out", "Strikeout"])]
        on_base = on_base.sort_values('count')

        pitch_types = on_base['CleanPitchType'].unique()

        col1, col2, col3 = st.columns(3)
        with st.container(border=True):
            with col3:
                top_pitches = st.number_input("Bottom N Pitches for On Base", min_value=3, max_value=len(pitch_types), value=5, step=1, format="%d")

        top_n_pitches = pitch_types[:top_pitches]
        last_pitch_result = last_pitch_result[last_pitch_result["CleanPitchType"].isin(top_n_pitches)]
        sorted_result = pd.concat([
            last_pitch_result[~last_pitch_result['BatterResult'].isin(["Out", "Strikeout"])].sort_values(by='count', ascending=True),
            last_pitch_result[last_pitch_result['BatterResult'].isin(["Out", "Strikeout"])]
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



with outs_by:
    # Select the appropriate 'mean_type' based on the selected pitcher team
    mean_type = ""
    if selected_pitcher_team == "All Teams":
        mean_type = "TeamName"
    else:
        mean_type = "PitcherId"

    st.markdown(f"### **Distributions of Strikes for {selected_pitcher_team} by {mean_type}**")

    # Aggregate data to get relevant columns
    data_clean = (
        data.groupby([mean_type, "GameID"])
        .agg(
            {
                "KorBB": lambda x: list(x),
                "PlayResult": lambda x: list(x),
                "CleanPitchCall": lambda x: list(x)
            }
        )
        .reset_index()
    )

    # Extract last values from lists in the columns
    data_clean["KorBB"] = data_clean["KorBB"].str.get(-1)
    data_clean["PlayResult"] = data_clean["PlayResult"].str.get(-1)

    # Determine BatterResult based on KorBB and PlayResult
    data_clean["BatterResult"] = np.where(
        data_clean["KorBB"].isin(["Strikeout", "Walk"]),
        data_clean["KorBB"],
        data_clean["PlayResult"],
    )

    data_clean[mean_type] = data_clean[mean_type].astype(dtype=str)

    # Exploding 'CleanPitchCall' column to have individual rows for each pitch call
    data_explode = data_clean.explode("CleanPitchCall")
    data_explode = data_explode[data_explode["CleanPitchCall"].isin(["Strike"])]

    # Create exploded DataFrame for the "Out" or "Strikeout" categories
    data_explode2 = (
        data_explode[
            (data_explode["CleanPitchCall"].isin(["Strike"]))
            & (data_explode["BatterResult"].isin(["Out", "Strikeout"]))
        ]
        .groupby([mean_type, "GameID"])
        .agg(count=("CleanPitchCall", "count"))
        .reset_index()
    )
    data_explode2["Only Caused Outs"] = "True"

    # Data cleaning for the second part (non-out) pitches
    data_explode = data_explode.groupby([mean_type, "GameID"]).agg(count=("CleanPitchCall", "count")).reset_index()
    data_explode["Only Caused Outs"] = "False"
    data_explode = pd.concat([data_explode, data_explode2])

    # Prepare data for plotting, adding a mean count per group
    dat_mean_type = data_explode.groupby([mean_type]).size().reset_index()
    dat_mean_type.columns = [mean_type, "count"]
    dat_mean_type = dat_mean_type.sort_values("count", ascending=False)

    # Select categories for plotting
    if mean_type == "TeamName":
        dat_mean_names = list(dat_mean_type[mean_type].unique())[:-1]
    else:
        dat_mean_names = list(dat_mean_type[mean_type].unique())[:5]

    # Filter data to include only selected mean names
    data_explode = data_explode[data_explode[mean_type].isin(dat_mean_names)]
    data_explode = data_explode.sort_values("count", ascending=False)

    # Create the ridgeline (violin) plot using Plotly Express
    fig = px.box(
        data_frame=data_explode,
        x="count",
        # y="Only Caused Outs",
        color="Only Caused Outs",
        boxmode="group",
        points="all",  # Show all points
        facet_col=mean_type,  # Facet by the mean_type (e.g., pitcher or team)
        facet_col_wrap=3,
        color_discrete_sequence=colors
    )

    # Customize layout to improve the appearance
    fig.update_layout(
        xaxis_title="",
        yaxis_title="Only Caused Outs",
        legend=dict(
            traceorder='reversed'  # This reverses the legend order
        ),
        yaxis=dict(
            range=[-.5, .33]  # Set the y-axis limits
        )
    )

    fig.update_xaxes(title_text="Count of Strikes Per Game", row=1, col=2)

    fig.update_yaxes(title_text="Only Caused Outs", row=3, col=1)

    if mean_type != "TeamName":
        fig.update_yaxes(title_text="Only Caused Outs", row=2, col=1)

    fig.for_each_annotation(
            lambda a: a.update(text=a.text.split("=")[-1], font=dict(color="black"))
        )

    # Display the plot in Streamlit
    st.plotly_chart(fig)


with appendix:
    st.title("Appendix")

    data_dictionary, test_cases, validity_of_data = st.tabs(["Data Dictionary", "Test Cases", "Validity of Data"])
    with data_dictionary:
        # Start HTML table
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

        # List of column names and descriptions
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
            ("CleanPitchCall", "Outcome of the pitch (Ball, Strike, etc.)."),
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
            ("PitcherTeam", "Team the pitcher is playing for."),
            ("HitSpinAxis", "Spin axis of ball after contact (degrees)."),
            ("Avg_Pitch_Speed", "Average pitch speed for the pitcher."),
            ("Avg_Vertical_Release_Angle", "Average vertical release angle for the pitcher."),
            ("Avg_Horizontal_Release_Angle", "Average horizontal release angle for the pitcher."),
            ("Avg_Spin_Rate", "Average spin rate for the pitcher."),
            ("Avg_Spin_Axis", "Average spin axis for the pitcher."),
            ("Strike_Percentage", "Percentage of pitches that were strikes."),
            ("Ball_Percentage", "Percentage of pitches that were balls."),
            ("Outs_Created", "Total outs created by the pitcher."),
            ("Avg_PlateLocHeight", "Average pitch height at the plate."),
            ("Avg_PlateLocSide", "Average pitch lateral location at the plate."),
            ("Pitch_Type_Diversity", "Measure of how many different pitch types a pitcher uses."),
            ("Max_Effective_Velocity", "Maximum effective velocity recorded."),
            ("Avg_Velocity_Drop", "Average drop in velocity from release to home plate."),
            ("Breaking_Ball_Ratio", "Ratio of breaking balls thrown compared to other pitches."),
            ("Pitch_Sequencing_Entropy", "Entropy measure of pitch sequencing strategy."),
            ("Pitch_Zonal_Targeting", "Strategy-based measure of pitch placement in the zone."),
            ("Fastball_to_Offspeed_Ratio", "Ratio of fastballs to offspeed pitches thrown."),
            ("Vertical_vs_Horizontal_Break_Ratio", "Ratio comparing vertical to horizontal pitch break."),
            ("Release_Extension_Deviation", "Deviation in release extension consistency."),
            ("Avg_Hit_Exit_Velocity", "Average exit velocity of batted balls."),
        ]

        # Append rows to the HTML table
        for column_name, description in data_dict:
            data_dict_html += f"<tr><td><b>{column_name}</b></td><td>{description}</td></tr>"

        # Close table
        data_dict_html += "</tbody></table>"

        # Streamlit App
        st.markdown("## Pioneer Baseball Data Dictionary")
        st.write("This table describes all key columns in the dataset:")

        # Display the table using Markdown
        st.markdown(data_dict_html, unsafe_allow_html=True)

    with test_cases:
        with open("Scripts/ML_Pipe/test_cases/testcase_result.md", "r") as file:
            md_content = file.read()

        st.markdown(md_content, unsafe_allow_html=True)

    
    with validity_of_data:
        df_valitity = data

        outs_df = df_valitity[df_valitity["OutsOnPlay"] > 0]

        # Define the pitch types to remove
        excluded_pitches = ["Knuckleball", "OneSeamFastBall", "Sweeper", "Other"]

        # Filter out these pitch types
        df_valitity = df_valitity[~df_valitity["CleanPitchType"].isin(excluded_pitches)]

        # Count total throws per pitch type
        total_pitches = df_valitity["CleanPitchType"].value_counts().reset_index()
        total_pitches.columns = ["CleanPitchType", "TotalPitches"]

        # Count pitches that resulted in a strike (called or swinging)
        strike_pitches = df_valitity[df_valitity["CleanPitchCall"].isin(["StrikeCalled", "StrikeSwinging"])]
        strike_counts = strike_pitches["CleanPitchType"].value_counts().reset_index()
        strike_counts.columns = ["CleanPitchType", "StrikePitches"]

        # Merge both counts
        pitch_strike_rates = total_pitches.merge(strike_counts, on="CleanPitchType", how="left").fillna(0)

        # Calculate the percentage of throws that resulted in a strike
        pitch_strike_rates["StrikePercentage"] = (pitch_strike_rates["StrikePitches"] / pitch_strike_rates["TotalPitches"]) * 100

        #Heatmap graph showing pitch location for outs
        fig = px.density_heatmap(outs_df, x="PlateLocSide", y="PlateLocHeight",
                                title="Pitch Location for Outs",
                                labels={"PlateLocSide": "Horizontal Location", "PlateLocHeight": "Vertical Location"},
                                nbinsx=30, nbinsy=30)

        st.plotly_chart(fig)

        #Scatter plot showing pitch speed and exit velocity 
        hits_df = df_valitity[df_valitity["PlayResult"].isin(["Single", "Double", "Triple", "HomeRun"])]

        fig = px.scatter(hits_df, x="RelSpeed", y="ExitSpeed", color="CleanPitchType",
                        title="Pitch Speed vs. Exit Velocity",
                        labels={"RelSpeed": "Pitch Speed (MPH)", "ExitSpeed": "Exit Velocity (MPH)"})

        st.plotly_chart(fig)


        # Filter for hit plays
        # Create a box plot instead of a scatter plot
        fig = px.box(hits_df, x="CleanPitchType", y="ExitSpeed", color="CleanPitchType",
                    title="Exit Velocity by Pitch Type",
                    labels={"CleanPitchType": "Pitch Type", "ExitSpeed": "Exit Velocity (MPH)"})

        st.plotly_chart(fig)


        runs_scored = df_valitity.groupby("Date")["RunsScored"].sum().reset_index()
        fig = px.line(runs_scored, x="Date", y="RunsScored", 
                    title="Runs Scored Over Time", 
                    labels={"RunsScored": "Total Runs", "Date": "Game Date"})
        st.plotly_chart(fig)


        col1,col2= st.columns(2)

        fig = px.box(df_valitity, y="RelHeight", title="Pitch Release Height Distribution",
                    labels={"RelHeight": "Release Height (feet)"}, color_discrete_sequence=["#5F7082"])
        
        with col1 : 
            st.plotly_chart(fig)




        pitches_per_inning = df_valitity.groupby(["GameID", "Inning"]).size().reset_index(name="Pitches")

        fig = px.box(pitches_per_inning, y="Pitches", title="Pitches Per Inning Distribution",
                    labels={"Pitches": "Pitches Per Inning"}, color_discrete_sequence=["#B4976B"])
        with col2 : 
            st.plotly_chart(fig)

        with future:
            st.title("The Future")
            st.markdown("### Accomplishments")
            st.markdown("- Data Analytics ")


            st.markdown("### Future Goals")
            st.markdown("- **Catcher Model** We want to create a model for the catcher")