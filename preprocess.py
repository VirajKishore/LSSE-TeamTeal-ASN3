import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import string
from collections import Counter
import json

from sklearn.feature_extraction.text import TfidfVectorizer

import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State, ALL
import plotly.graph_objects as go
import dash_bootstrap_components as dbc



# ---------------------------------------------------------
# 1. Data Loading (Feature Releases)
# ---------------------------------------------------------
df_releases = pd.read_excel("Firefox rf.xlsx", engine="openpyxl")
df_releases["Release Date"] = pd.to_datetime(df_releases["Release Date"]).dt.date
release_counts_series = df_releases.groupby("Release Date").size()
release_counts = release_counts_series.to_dict()

years = sorted({d.year for d in release_counts.keys()})
if not years:
    years = [datetime.now().year]
global_max = max(release_counts.values()) if release_counts else 0

# ---------------------------------------------------------
# 2. Data Loading (Reviews)
# ---------------------------------------------------------
df_reviews_raw = pd.read_excel("Firefox ar.xlsx", engine="openpyxl")
df_reviews_raw["at"] = pd.to_datetime(df_reviews_raw["at"], errors="coerce")
df_reviews_raw["score"] = pd.to_numeric(df_reviews_raw["score"], errors="coerce")

# ---------------------------------------------------------
# 3. Top Keywords Extraction
# ---------------------------------------------------------
def get_top_keywords_tfidf(text, n=10, category=None):
    base_stopwords = {
        "the", "and", "is", "to", "of", "in", "a", "for", "it", "that",
        "on", "with", "as", "its", "this", "i", "you", "we", "are", "was",
        "firefox", "browser", "good", "app", "android", "my", "be", "best"
    }
    if category in ["neutral", "bad"]:
        extra_stopwords = {"ok", "fine", "average", "mediocre", "not", "no", "but", "just", "really"}
    else:
        extra_stopwords = set()
    stopwords = base_stopwords.union(extra_stopwords)

    translator = str.maketrans("", "", string.punctuation)
    cleaned_text = text.translate(translator).lower()
    if not cleaned_text.strip():
        return ""
    vectorizer = TfidfVectorizer(stop_words=stopwords, token_pattern=r"(?u)\b\w\w+\b")
    try:
        tfidf_matrix = vectorizer.fit_transform([cleaned_text])
    except ValueError:
        return ""
    feature_names = vectorizer.get_feature_names_out()
    if len(feature_names) == 0:
        return ""
    scores = tfidf_matrix.toarray().flatten()
    sorted_indices = np.argsort(scores)[::-1]
    top_keywords = [feature_names[i] for i in sorted_indices if scores[i] > 0][:n]
    return ", ".join(top_keywords)

def get_top_keywords(text, n=10, category=None):
    keywords = get_top_keywords_tfidf(text, n, category)
    if not keywords:
        translator = str.maketrans("", "", string.punctuation)
        words = text.translate(translator).lower().split()
        fallback_stopwords = {"the", "and", "is", "to", "of", "in", "a", "for", "it", "that", "i", "me", "be", "at", "by", "on", "or", "but"}
        words = [w for w in words if w not in fallback_stopwords]
        if not words:
            return "No keywords"
        freq = Counter(words)
        top = freq.most_common(n)
        if not top:
            return "No keywords"
        return ", ".join(word for word, cnt in top)
    else:
        return keywords

# ---------------------------------------------------------
# 4. Weekly Aggregation for Reviews
# ---------------------------------------------------------
def aggregate_week(df_week):
    good_count = (df_week["score"] >= 4).sum()
    neutral_count = (df_week["score"] == 3).sum()
    bad_count = (df_week["score"] <= 2).sum()
    avg_score = df_week["score"].mean()

    content_all = " ".join(df_week["content"].dropna().astype(str))
    content_good = " ".join(df_week.loc[df_week["score"] >= 4, "content"].dropna().astype(str))
    content_neutral = " ".join(df_week.loc[df_week["score"] == 3, "content"].dropna().astype(str))
    content_bad = " ".join(df_week.loc[df_week["score"] <= 2, "content"].dropna().astype(str))

    top_all = get_top_keywords(content_all, n=10, category="all")
    top_good = get_top_keywords(content_good, n=10, category="good")
    top_neutral = get_top_keywords(content_neutral, n=10, category="neutral")
    top_bad = get_top_keywords(content_bad, n=10, category="bad")

    return pd.Series({
        "good": good_count,
        "neutral": neutral_count,
        "bad": bad_count,
        "average": avg_score,
        "top_all": top_all,
        "top_good": top_good,
        "top_neutral": top_neutral,
        "top_bad": top_bad,
        "content_all": content_all,
        "content_good": content_good,
        "content_neutral": content_neutral,
        "content_bad": content_bad
    })

def get_discrete_weeks_for_year(selected_year, monthly=False, selected_month=None):
    if not monthly:
        d_start = date(selected_year, 1, 1)
        d_end = date(selected_year, 12, 31)
    else:
        d_start = date(selected_year, selected_month, 1)
        if selected_month < 12:
            d_end = date(selected_year, selected_month + 1, 1) - timedelta(days=1)
        else:
            d_end = date(selected_year, 12, 31)
    calendar_start = d_start - timedelta(days=d_start.weekday())
    calendar_end = d_end + timedelta(days=(6 - d_end.weekday()))
    total_days = (calendar_end - calendar_start).days + 1
    num_weeks = total_days // 7
    all_dates = [calendar_start + timedelta(days=i) for i in range(total_days)]
    return d_start, d_end, calendar_start, calendar_end, total_days, num_weeks, all_dates

def aggregate_reviews_for_discrete_weeks(selected_year, monthly=False, selected_month=None):
    d_start, d_end, cal_start, _, total_days, num_weeks, _ = get_discrete_weeks_for_year(selected_year, monthly, selected_month)
    df_rev = df_reviews_raw.copy()
    df_rev = df_rev[df_rev["at"].dt.date.between(d_start, d_end)]
    if df_rev.empty:
        return pd.DataFrame(), num_weeks
    df_rev["Monday"] = df_rev["at"].dt.date - pd.to_timedelta(df_rev["at"].dt.weekday, unit="D")
    grouped = df_rev.groupby("Monday").apply(aggregate_week).reset_index()

    def monday_to_week(monday_date):
        delta = (monday_date - cal_start).days
        return delta // 7 + 1

    grouped["x_week"] = grouped["Monday"].apply(monday_to_week)
    return grouped, num_weeks

# ---------------------------------------------------------
# 5. Discrete Calendar Heatmap & Reviews Chart
# ---------------------------------------------------------
def generate_discrete_calendar_heatmap(selected_year, release_range, monthly=False, selected_month=None):
    day_map = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    d_start, d_end, cal_start, cal_end, total_days, num_weeks, all_dates = get_discrete_weeks_for_year(selected_year, monthly, selected_month)

    z = np.full((7, num_weeks), 0.0)
    day_matrix = np.full((7, num_weeks), '', dtype=object)
    custom_data = np.full((7, num_weeks), '', dtype=object)

    for i, current_date in enumerate(all_dates):
        week_idx = i // 7
        day_idx = current_date.weekday()
        cnt = release_counts.get(current_date, 0)
        if d_start <= current_date <= d_end:
            day_matrix[day_idx, week_idx] = str(current_date.day)
            custom_data[day_idx, week_idx] = current_date.strftime("%Y-%m-%d")
        if cnt == 0:
            z[day_idx, week_idx] = 0
        elif release_range[0] <= cnt <= release_range[1]:
            z[day_idx, week_idx] = cnt
        else:
            z[day_idx, week_idx] = np.nan

    heatmap_trace = go.Heatmap(
        z=z,
        x=list(range(1, num_weeks+1)),
        y=day_map,
        text=day_matrix,
        texttemplate="%{text}",
        textfont=dict(size=10, color="black"),
        customdata=custom_data,
        hovertemplate="Date: %{customdata}<br>Count: %{z}<extra></extra>",
        colorscale=[
            [0.0, "#ffffff"],
            [0.00001, "#deebf7"],
            [0.2, "#c6dbef"],
            [0.4, "#9ecae1"],
            [0.6, "#6baed6"],
            [0.8, "#4292c6"],
            [1.0, "#2171b5"]
        ],
        colorbar=dict(title="Release Count"),
        zmin=0,
        zmax=global_max,
        xgap=3,
        ygap=3,
        name="Daily Releases"
    )

    # Month labels
    months_x, months_y, months_text, months_customdata = [], [], [], []
    for m in range(1, 13):
        try:
            first_day_of_month = date(selected_year, m, 1)
        except ValueError:
            continue
        if not (d_start <= first_day_of_month <= d_end):
            continue
        offset = (first_day_of_month - cal_start).days
        week_for_month = offset // 7
        x_val = week_for_month + 1
        y_val = 7.5
        months_x.append(x_val)
        months_y.append(y_val)
        months_text.append(first_day_of_month.strftime("%b"))
        months_customdata.append(m)

    month_label_trace = go.Scatter(
        x=months_x,
        y=months_y,
        text=months_text,
        customdata=months_customdata,
        mode="text",
        hoverinfo="none",
        showlegend=False,
        name="Month Labels",
        textfont=dict(size=14, color="black")
    )

    title_str = f"Feature Releases Calendar Heat Map - {selected_year}"
    if monthly:
        title_str += f" (Month {selected_month})"

    fig = go.Figure(data=[heatmap_trace, month_label_trace])
    fig.update_layout(
        title=title_str,
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickmode="array",
            tickvals=list(range(1, num_weeks+1)),
            ticktext=[f"W{i}" for i in range(1, num_weeks+1)]
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        height=400,
        font=dict(family="Arial")
    )
    return fig

def generate_discrete_reviews_linechart(selected_year, monthly=False, selected_month=None):
    agg, num_weeks = aggregate_reviews_for_discrete_weeks(selected_year, monthly, selected_month)
    if agg.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"Reviews - No Data for {selected_year}",
            xaxis=dict(tickvals=[], ticktext=[]),
            height=300
        )
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=agg["x_week"], y=agg["good"],
        mode="lines+markers", name="Good Reviews",
        customdata=agg["top_good"],
        hovertemplate="Week %{x}<br>Good Reviews: %{y}<br>Top Words: %{customdata}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=agg["x_week"], y=agg["neutral"],
        mode="lines+markers", name="Neutral Reviews",
        customdata=agg["top_neutral"],
        hovertemplate="Week %{x}<br>Neutral Reviews: %{y}<br>Top Words: %{customdata}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=agg["x_week"], y=agg["bad"],
        mode="lines+markers", name="Bad Reviews",
        customdata=agg["top_bad"],
        hovertemplate="Week %{x}<br>Bad Reviews: %{y}<br>Top Words: %{customdata}<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        x=agg["x_week"], y=agg["average"],
        mode="lines+markers", name="Average Score", yaxis="y2",
        customdata=agg["top_all"],
        hovertemplate="Week %{x}<br>Average Score: %{y:.2f}<br>Top Words: %{customdata}<extra></extra>"
    ))

    fig.update_layout(
        title=f"Weekly Reviews - {selected_year}",
        xaxis=dict(
            title="Discrete Week Index",
            tickmode="array",
            tickvals=list(range(1, num_weeks+1)),
            ticktext=[f"W{i}" for i in range(1, num_weeks+1)]
        ),
        yaxis=dict(title="Review Counts"),
        yaxis2=dict(title="Average Score", overlaying="y", side="right", range=[0,5]),
        legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
        margin=dict(l=40, r=40, t=60, b=40),
        height=300,
        font=dict(family="Arial")
    )
    return fig

def generate_two_discrete_graphs(selected_year, view_type, selected_month, release_range):
    if view_type == "monthly":
        reviews_fig = generate_discrete_reviews_linechart(selected_year, monthly=True, selected_month=selected_month)
        calendar_fig = generate_discrete_calendar_heatmap(selected_year, release_range, monthly=True, selected_month=selected_month)
    else:
        reviews_fig = generate_discrete_reviews_linechart(selected_year, monthly=False)
        calendar_fig = generate_discrete_calendar_heatmap(selected_year, release_range, monthly=False)
    return reviews_fig, calendar_fig

# ---------------------------------------------------------
# 6. Main App Layout (Two-Row Controls + Two Equal-Width Data Divs)
# ---------------------------------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server 
app.title = "Discrete Weeks Calendar & Reviews"

app.layout = dbc.Container(fluid=True, children=[
    # Title
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    html.H1("Discrete Weeks Calendar & Reviews", className="text-center mb-0")
                ),
                className="my-4 shadow border-0",
                style={"borderRadius": "15px"}
            ),
            width=12
        )
    ]),
    # Controls Row #1
    dbc.Row([
    dbc.Col([
        html.Label("Select a Year:", className="fw-bold mb-1"),
        dcc.Dropdown(
            id="year-dropdown",
            options=[{"label": str(y), "value": y} for y in years],
            value=years[0],
            clearable=False,
            style={"width": "100%"}
        )
    ], xs=12, sm=6, md=3),

    dbc.Col([
        html.Label("View:", className="fw-bold mb-1"),
        dcc.RadioItems(
            id="view-type",
            options=[
                {"label": "Yearly", "value": "yearly"},
                {"label": "Monthly", "value": "monthly"}
            ],
            value="yearly",
            labelStyle={"display": "inline-block", "marginRight": "15px"},
            inputStyle={"marginRight": "5px"}
        )
    ], xs=12, sm=6, md=3),

    dbc.Col([
        html.Label("Filter by Release Count:", className="fw-bold mb-1"),
        dcc.RangeSlider(
            id="release-range",
            min=0,
            max=global_max,
            step=1,
            value=[0, global_max],
            tooltip={"placement": "bottom"},
            marks={i: str(i) for i in range(0, global_max + 1)}
        )
    ], xs=12, md=4),

    dbc.Col([
        html.Label(" ", className="mb-1"),  # spacer label
        dbc.Button("Reset", id="reset-button", color="secondary", n_clicks=0, className="w-100")
    ], xs=12, md=2)
], className="g-3 mb-4"),


    # Controls Row #2 (Month dropdown)
    dbc.Row([
        dbc.Col(
            html.Div(
                html.Div([
                    html.Label("Month:", className="mr-2"),
                    dcc.Dropdown(
                        id="month-dropdown",
                        options=[
                            {"label": "January", "value": 1},
                            {"label": "February", "value": 2},
                            {"label": "March", "value": 3},
                            {"label": "April", "value": 4},
                            {"label": "May", "value": 5},
                            {"label": "June", "value": 6},
                            {"label": "July", "value": 7},
                            {"label": "August", "value": 8},
                            {"label": "September", "value": 9},
                            {"label": "October", "value": 10},
                            {"label": "November", "value": 11},
                            {"label": "December", "value": 12},
                        ],
                        value=1,
                        clearable=False,
                        style={"width": "150px"}
                    )
                ], style={"display": "inline-flex", "alignItems": "center"})
            ),
            id="month-dropdown-container",
            style={"display": "none"},
            md=12
        )
    ], className="mb-3"),

    # Reviews Graph
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(
                        id="reviews-graph",
                        config={"displayModeBar": False},
                        style={"width": "100%"}
                    )
                ),
                className="mb-4 shadow border-0",
                style={"borderRadius": "15px"}
            ),
            width=12
        )
    ]),
    # Calendar Heatmap
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(
                        id="calendar-graph",
                        config={"displayModeBar": False},
                        style={"width": "100%"}
                    )
                ),
                className="mb-4 shadow border-0",
                style={"borderRadius": "15px"}
            ),
            width=12
        )
    ]),

    # Two equal-width columns for software releases & keywords/review expansions
    dbc.Row([
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Software Releases", className="mb-3"),
                    html.Div(id="day-details", children="Click on a day tile to see feature updates.")
                ]),
                className="shadow border-0 mb-4",
                style={"borderRadius": "15px", "height": "500px", "overflowY": "auto"}
            ),
            md=6, sm=12, xs=12
        ),
        dbc.Col(
            dbc.Card(
                dbc.CardBody([
                    html.H5("Top Keywords / Expanded Reviews", className="mb-3"),
                    html.Div(id="keywords-div", className="mb-3"),
                    html.Hr(),
                    html.Div(id="reviews-details-div")
                ]),
                className="shadow border-0 mb-4",
                style={"borderRadius": "15px", "height": "500px", "overflowY": "auto"}
            ),
            md=6, sm=12, xs=12
        )
    ]),

    # Hidden store
    dcc.Store(id="selected-review-store")
], style={"backgroundColor": "#f8f9fa", "padding": "20px"})

# ---------------------------------------------------------
# 7. Callbacks
# ---------------------------------------------------------
@app.callback(
    Output("month-dropdown-container", "style"),
    Input("view-type", "value")
)
def toggle_month_dropdown(view_type):
    if view_type == "monthly":
        return {"display": "block"}
    return {"display": "none"}

@app.callback(
    Output("reviews-graph", "figure"),
    Output("calendar-graph", "figure"),
    Output("day-details", "children"),
    Input("reviews-graph", "clickData"),
    Input("calendar-graph", "clickData"),
    Input("year-dropdown", "value"),
    Input("view-type", "value"),
    Input("month-dropdown", "value"),
    Input("release-range", "value"),
    Input("reset-button", "n_clicks")
)
def update_main_graphs(revClick, calClick, selected_year, view_type, selected_month, release_range, reset_clicks):
    monthly_flag = (view_type == "monthly")
    reviews_fig, calendar_fig = generate_two_discrete_graphs(selected_year, view_type, selected_month, release_range)

    day_details = "Click on a day tile to see feature updates."
    triggered = callback_context.triggered
    if triggered:
        source = triggered[0]["prop_id"].split(".")[0]

        # If the reviews graph is clicked => show releases for the entire week
        if source == "reviews-graph" and revClick:
            week = revClick["points"][0]["x"]
            d_start, d_end, cal_start, cal_end, total_days, num_weeks, _ = get_discrete_weeks_for_year(selected_year, monthly_flag, selected_month)
            monday_date = cal_start + timedelta(weeks=week - 1)
            sunday_date = monday_date + timedelta(days=6)
            df_week_releases = df_releases[df_releases["Release Date"].between(monday_date, sunday_date)]
            if df_week_releases.empty:
                day_details = f"No releases found for week {week} ({monday_date} to {sunday_date})."
            else:
                items = [html.Li(row["Feature Description"]) for _, row in df_week_releases.iterrows()]
                day_details = html.Div([
                    html.H5(f"Software Releases for Week {week} ({monday_date} to {sunday_date}):", className="mb-2"),
                    html.Ul(items, style={"marginLeft": "20px", "textAlign": "left"})
                ])

        # If the calendar is clicked => show releases for that specific day
        elif source == "calendar-graph" and calClick:
            date_str = calClick["points"][0].get("customdata", "")
            if date_str:
                try:
                    clicked_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                except Exception:
                    return reviews_fig, calendar_fig, "Unable to parse clicked date."
                df_day_releases = df_releases[df_releases["Release Date"] == clicked_date]
                if df_day_releases.empty:
                    day_details = f"No releases found on {clicked_date}"
                else:
                    items = [html.Li(row["Feature Description"]) for _, row in df_day_releases.iterrows()]
                    day_details = html.Div([
                        html.H5(f"Software Releases on {clicked_date.strftime('%b %d, %Y')}:", className="mb-2"),
                        html.Ul(items, style={"marginLeft": "20px", "textAlign": "left"})
                    ])

    return reviews_fig, calendar_fig, day_details

@app.callback(
    Output("selected-review-store", "data"),
    Input("reviews-graph", "clickData"),
    State("year-dropdown", "value"),
    State("view-type", "value"),
    State("month-dropdown", "value")
)
def update_selected_review_store(revClick, selected_year, view_type, selected_month):
    if not revClick:
        return {}
    point = revClick["points"][0]
    week = point["x"]
    curve_num = point["curveNumber"]
    cat_map = {0:"good", 1:"neutral", 2:"bad", 3:"all"}
    category = cat_map.get(curve_num, "good")
    monthly_flag = (view_type == "monthly")
    agg, _ = aggregate_reviews_for_discrete_weeks(selected_year, monthly_flag, selected_month)
    row = agg[agg["x_week"] == week]
    if row.empty:
        return {}
    row = row.iloc[0]
    if category == "good":
        keywords = row["top_good"]
        content = row["content_good"]
    elif category == "neutral":
        keywords = row["top_neutral"]
        content = row["content_neutral"]
    elif category == "bad":
        keywords = row["top_bad"]
        content = row["content_bad"]
    else:
        keywords = row["top_all"]
        content = row["content_all"]
    return {"week": int(week), "category": category, "keywords": keywords, "content": content, "year": selected_year}

@app.callback(
    Output("keywords-div", "children"),
    Input("selected-review-store", "data")
)
def update_keywords_div(data):
    if not data or "keywords" not in data or not data["keywords"]:
        return html.Div("Click a point in the reviews chart to see top keywords for that week.")
    keywords_str = data["keywords"]
    if keywords_str == "No keywords" or not keywords_str.strip():
        return html.Div("No keywords found for this category/week.")
    words = [w.strip() for w in keywords_str.split(",") if w.strip()]
    if not words:
        return html.Div("No keywords found for this category/week.")

    buttons = [
        dbc.Button(word, id={"type": "keyword-btn", "index": word}, color="info", className="m-1")
        for word in words
    ]
    return html.Div([
        html.H6(f"Top {len(words)} Keywords for Week {data.get('week')} ({data.get('category').capitalize()} Reviews):"),
        html.Div(buttons, style={"display": "flex", "flexWrap": "wrap"})
    ])

@app.callback(
    Output("reviews-details-div", "children"),
    Input({"type": "keyword-btn", "index": ALL}, "n_clicks"),
    State("selected-review-store", "data"),
    prevent_initial_call=True
)
def show_keyword_reviews(n_clicks_list, store_data):
    ctx = callback_context
    if not ctx.triggered or not store_data:
        return ""
    triggered = ctx.triggered[0]
    if "keyword-btn" not in triggered["prop_id"]:
        return ""
    button_id = json.loads(triggered["prop_id"].split(".")[0])
    clicked_keyword = button_id["index"]

    selected_week = store_data["week"]
    selected_year = store_data["year"]
    category = store_data["category"]

    d_start, d_end, cal_start, cal_end, total_days, num_weeks, _ = get_discrete_weeks_for_year(selected_year)
    monday_date = cal_start + timedelta(weeks=selected_week - 1)
    week_end = monday_date + timedelta(days=6)
    df_week = df_reviews_raw[df_reviews_raw["at"].dt.date.between(monday_date, week_end)]
    if category == "good":
        df_week = df_week[df_week["score"] >= 4]
    elif category == "neutral":
        df_week = df_week[df_week["score"] == 3]
    elif category == "bad":
        df_week = df_week[df_week["score"] <= 2]

    mask = df_week["content"].str.lower().str.contains(clicked_keyword.lower(), na=False)
    df_filtered = df_week[mask]
    if df_filtered.empty:
        return dbc.Alert(f"No reviews found containing '{clicked_keyword}' in week {selected_week}.", color="warning")

    reviews_list = df_filtered["content"].head(5).tolist()
    return html.Div([
        html.H6(f"Reviews containing '{clicked_keyword}' in week {selected_week}:"),
        html.Ul([html.Li(review) for review in reviews_list], style={"maxHeight": "200px", "overflowY": "auto"})
    ])

# ---------------------------------------------------------
# 7. Run the App
# ---------------------------------------------------------


if __name__ == "__main__":
    app.run_server(debug=False)


