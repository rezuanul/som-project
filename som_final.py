import pandas as pd
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# ─── 1) Load datasets ────────────────────────────────────────────────────────────
df = pd.read_csv("Updated_Dataset_with_Sell_Status.csv")
df.drop(columns=["Scope_and_Focus"], errors="ignore", inplace=True)

df_operational = pd.read_csv("Updated_Dataset_with_Operational_Inputs.csv")
df_operational.drop(columns=["Scope_and_Focus"], errors="ignore", inplace=True)

# ─── 2) Compute per-kg CO₂-eq ────────────────────────────────────────────────────
impact_factors = {
    "Cotton Waste": 18.9,
    "Synthetic Fiber Waste": 4.4,
    "Thread Waste":  4.4,
    "Packaging Material": 3.0,
    "Denim Scraps": 20.0,
    "Chemical Containers": 1.5
}
df_operational["CO2_kg"] = (
    df_operational["Waste_Quantity_Kg"] *
    df_operational["Waste_Type"].map(impact_factors).fillna(0)
)

# ─── 3) Month ordering & future years ───────────────────────────────────────────
month_order = [
    "January","February","March","April","May","June",
    "July","August","September","October","November","December"
]
df["Month"]             = pd.Categorical(df["Month"], categories=month_order, ordered=True)
df_operational["Month"] = pd.Categorical(df_operational["Month"], categories=month_order, ordered=True)
future_years = [2026, 2027]

# ─── 4) Train “what-if” models ───────────────────────────────────────────────────
features = ["Company", "Cutting_Technique", "Sorting_Process", "Worker_Experience_Years"]
X_rq2 = pd.get_dummies(df_operational[features], drop_first=True)
y_rq2 = df_operational["Waste_Quantity_Kg"]

models_rq2 = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "Linear": LinearRegression()
}
for mdl in models_rq2.values():
    mdl.fit(X_rq2, y_rq2)

# ─── 5) ==== NEW: Model Validation (80/20 split + metrics) ==== ────────────────
# Prepare train/test
X = X_rq2.copy()
y = y_rq2.copy()
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2,
                                                    random_state=42)

# Fit & evaluate
eval_rows = []
for name, mdl in models_rq2.items():
    mdl.fit(X_train, y_train)
    preds = mdl.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)
    eval_rows += [
        {"Model": name, "Metric": "RMSE", "Value": rmse},
        {"Model": name, "Metric": "MAE",  "Value": mae},
        {"Model": name, "Metric": "R²",   "Value": r2},
    ]
df_metrics = pd.DataFrame(eval_rows)

# Build the evaluation figure
fig_eval = px.scatter(
    df_metrics,
    x="Model", y="Value",
    color="Metric", symbol="Metric",
    title="Model Validation Metrics",
    facet_col="Metric", facet_col_wrap=3,
    labels={"Value": "Metric Value"}
)
fig_eval.update_yaxes(matches=None)

# ─── 6) Initialize Dash ─────────────────────────────────────────────────────────
app = dash.Dash(__name__)
app.title = "Textile Waste Monitoring & LCA Dashboard"

# ─── 7) App layout ─────────────────────────────────────────────────────────────
app.layout = html.Div([

    # === 1) Company-Wise Textile Waste Performance ===
    html.H2("📊 Company-Wise Textile Waste Performance", style={"textAlign":"center"}),
    html.Label("Select Company:"),
    dcc.Dropdown(
        id="company-dropdown",
        options=[{"label":c,"value":c} for c in df["Company"].unique()],
        value=df["Company"].unique()[0],
        style={"width":"50%"}
    ),
    html.H4("Filtered Yearly Waste Quantity by Type"),
    html.Div([
        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id="filter-year",
                options=[{"label":"All","value":"All"}] +
                        [{"label":str(y),"value":y} for y in sorted(df["Year"].unique())],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block"}),
        html.Div([
            html.Label("Month:"),
            dcc.Dropdown(
                id="filter-month",
                options=[{"label":"All","value":"All"}] +
                        [{"label":m,"value":m} for m in month_order],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"2%"}),
        html.Div([
            html.Label("Waste Type:"),
            dcc.Dropdown(
                id="filter-waste",
                options=[{"label":"All","value":"All"}] +
                        [{"label":w,"value":w} for w in sorted(df["Waste_Type"].unique())],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"2%"})
    ]),
    dcc.Graph(id="yearly-waste-type"),
    dcc.Graph(id="monthly-recycle-use"),

    # === 2) Monthly Recycle Use Trends (Filtered) ===
    html.H4("♻️ Monthly Recycle Use Trends (Filtered)", style={"textAlign":"center"}),
    html.Div([
        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id="year-filter",
                options=[{"label":"All","value":"All"}] +
                        [{"label":str(y),"value":y} for y in sorted(df["Year"].unique())],
                value="All"
            )
        ], style={"width":"30%","display":"inline-block"}),
        html.Div([
            html.Label("Month:"),
            dcc.Dropdown(
                id="month-filter",
                options=[{"label":"All","value":"All"}] +
                        [{"label":m,"value":m} for m in month_order],
                value="All"
            )
        ], style={"width":"30%","display":"inline-block","marginLeft":"5%"}),
        html.Div([
            html.Label("Company:"),
            dcc.Dropdown(
                id="company-filter",
                options=[{"label":"All","value":"All"}] +
                        [{"label":c,"value":c} for c in sorted(df["Company"].unique())],
                value="All"
            )
        ], style={"width":"30%","display":"inline-block","marginLeft":"5%"})
    ], style={"marginBottom":"20px"}),
    dcc.Graph(id="monthly-recycle-use-filtered"),

    # === 3) Environmental Impact Filter View ===
    html.H4("🌍 Environmental Impact Filter View", style={"textAlign":"center"}),
    html.Div([
        html.Div([
            html.Label("Company:"),
            dcc.Dropdown(
                id="impact-company",
                options=[{"label":"All","value":"All"}] +
                        [{"label":i,"value":i} for i in sorted(df["Company"].unique())],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block"}),
        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id="impact-year",
                options=[{"label":"All","value":"All"}] +
                        [{"label":str(y),"value":y} for y in sorted(df["Year"].unique())],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"2%"}),
        html.Div([
            html.Label("Month:"),
            dcc.Dropdown(
                id="impact-month",
                options=[{"label":"All","value":"All"}] +
                        [{"label":m,"value":m} for m in month_order],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"2%"}),
        html.Div([
            html.Label("Waste Type:"),
            dcc.Dropdown(
                id="impact-waste",
                options=[{"label":"All","value":"All"}] +
                        [{"label":w,"value":w} for w in sorted(df["Waste_Type"].unique())],
                value="All"
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"2%"})
    ], style={"marginBottom":"20px"}),
    dcc.Graph(id="impact-custom-graph"),
    dcc.Graph(id="impact-comparison"),

    # === 4) Company Waste Contribution ===
    html.H4("🧾 Company Waste Contribution", style={"textAlign":"center"}),
    html.Div([
        html.Div([
            html.Label("Waste Type:"),
            dcc.Dropdown(
                id="waste-comparison-type",
                options=[{"label":"All","value":"All"}] +
                        [{"label":w,"value":w} for w in sorted(df["Waste_Type"].unique())],
                value="All"
            )
        ], style={"width":"48%","display":"inline-block"}),
        html.Div([
            html.Label("Year:"),
            dcc.Dropdown(
                id="waste-comparison-year",
                options=[{"label":"All","value":"All"}] +
                        [{"label":str(y),"value":y} for y in sorted(df["Year"].unique())],
                value="All"
            )
        ], style={"width":"48%","display":"inline-block","float":"right"})
    ]),
    dcc.Graph(id="company-waste-comparison"),

    # === 5) Ready for Sell Inventory ===
    html.H3("🔍 Ready for Sell Inventory", style={"textAlign":"center"}),
    html.Div([
        html.Div([
            html.Label("Waste Type (single):"),
            dcc.Dropdown(
                id="waste-type-single",
                options=[{"label":"All","value":"All"}] +
                        [{"label":w,"value":w} for w in sorted(df["Waste_Type"].unique())],
                value="All"
            ),
            html.Br(),
            html.Label("Waste Type (multi):"),
            dcc.Dropdown(
                id="waste-type-multi",
                options=[{"label":w,"value":w} for w in sorted(df["Waste_Type"].unique())],
                multi=True
            )
        ], style={"width":"45%","display":"inline-block"}),
        html.Div([
            html.Label("Recycle Use (single):"),
            dcc.Dropdown(
                id="recycle-type-single",
                options=[{"label":"All","value":"All"}] +
                        [{"label":r,"value":r} for r in sorted(df["Recycle_Use"].unique())],
                value="All"
            ),
            html.Br(),
            html.Label("Recycle Use (multi):"),
            dcc.Dropdown(
                id="recycle-type-multi",
                options=[{"label":r,"value":r} for r in sorted(df["Recycle_Use"].unique())],
                multi=True
            )
        ], style={"width":"45%","display":"inline-block","float":"right"})
    ], style={"marginBottom":"20px"}),
    html.Div(id="ready-items-output"),

    # === 6) Economic vs Environmental Trade-Off (Rating) ===
    html.H2("♻️ Economic vs Environmental Trade-Off (Rating)", style={"textAlign":"center"}),
    html.Div([
        dcc.Dropdown(
            id="tradeoff-company",
            options=[{"label":"All","value":"All"}] +
                    [{"label":c,"value":c} for c in sorted(df_operational["Company"].unique())],
            value="All",
            style={"width":"30%"}
        ),
        dcc.Dropdown(
            id="tradeoff-year",
            options=[{"label":"All","value":"All"}] +
                    [{"label":str(y),"value":y} for y in sorted(df_operational["Year"].unique())],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        ),
        dcc.Dropdown(
            id="tradeoff-month",
            options=[{"label":"All","value":"All"}] +
                    [{"label":m,"value":m} for m in month_order],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        )
    ], style={"marginBottom":"20px"}),
    dcc.Graph(id="tradeoff-graph"),
    html.Div(id="summary-metrics", style={"padding":"20px"}),

    # === 7) Performance & CO₂-eq Impact ===
    html.H2("♻️ Performance & CO₂-eq Impact", style={"textAlign":"center"}),
    html.Div([
        dcc.Dropdown(
            id="perf-company",
            options=[{"label":c,"value":c} for c in sorted(df_operational["Company"].unique())],
            value=df_operational["Company"].unique()[0],
            style={"width":"30%"}
        ),
        dcc.Dropdown(
            id="perf-year",
            options=[{"label":"All","value":"All"}] +
                    [{"label":str(y),"value":y} for y in sorted(df_operational["Year"].unique())],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        ),
        dcc.Dropdown(
            id="perf-month",
            options=[{"label":"All","value":"All"}] +
                    [{"label":m,"value":m} for m in month_order],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        )
    ], style={"marginBottom":"20px"}),
    dcc.Graph(id="yearly-waste-fig"),
    dcc.Graph(id="yearly-co2-fig"),

    # === 8) Trade-Off (CO₂) ===
    html.H2("💰 vs 🌍 Trade-Off (CO₂-eq)", style={"textAlign":"center"}),
    html.Div([
        dcc.Dropdown(
            id="trade-company",
            options=[{"label":"All","value":"All"}] +
                    [{"label":c,"value":c} for c in sorted(df_operational["Company"].unique())],
            value="All",
            style={"width":"30%"}
        ),
        dcc.Dropdown(
            id="trade-year",
            options=[{"label":"All","value":"All"}] +
                    [{"label":str(y),"value":y} for y in sorted(df_operational["Year"].unique())],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        ),
        dcc.Dropdown(
            id="trade-month",
            options=[{"label":"All","value":"All"}] +
                    [{"label":m,"value":m} for m in month_order],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        )
    ], style={"marginBottom":"20px"}),
    dcc.Graph(id="tradeoff-scatter"),
    html.Div(id="tradeoff-summary", style={"padding":"10px"}),

    # === 9) Forecast Comparison by Model ===
    html.H2("🔮 Forecast Comparison by Model", style={"textAlign":"center"}),
    html.Div([
        dcc.Dropdown(
            id="pred-company",
            options=[{"label":"All","value":"All"}] +
                    [{"label":c,"value":c} for c in sorted(df_operational["Company"].unique())],
            value="All",
            style={"width":"30%"}
        ),
        dcc.Dropdown(
            id="pred-month",
            options=[{"label":"All","value":"All"}] +
                    [{"label":m,"value":m} for m in month_order],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        ),
        dcc.Dropdown(
            id="pred-year",
            options=[{"label":"All","value":"All"}] +
                    [{"label":str(y),"value":y} for y in future_years],
            value="All",
            style={"width":"20%","marginLeft":"5px","display":"inline-block"}
        )
    ], style={"marginBottom":"20px"}),
    dcc.Graph(id="forecast-waste-compare"),
    dcc.Graph(id="forecast-recycle-compare"),
    dcc.Graph(id="forecast-co2-compare"),

    html.Hr(),

    # === 10) What-If Operational Waste Prediction ===
    html.H2("🔍 What-If Operational Waste Prediction", style={"textAlign":"center"}),
    html.Div([
        html.Div([
            html.Label("Company"),
            dcc.Dropdown(
                id="rq2-company",
                options=[{"label":c,"value":c} for c in sorted(df_operational["Company"].unique())],
                value=sorted(df_operational["Company"].unique())[0]
            )
        ], style={"width":"24%","display":"inline-block"}),
        html.Div([
            html.Label("Cutting Technique"),
            dcc.Dropdown(
                id="rq2-cutting",
                options=[{"label":t,"value":t} for t in sorted(df_operational["Cutting_Technique"].unique())],
                value=sorted(df_operational["Cutting_Technique"].unique())[0]
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"1%"}),
        html.Div([
            html.Label("Sorting Process"),
            dcc.Dropdown(
                id="rq2-sorting",
                options=[{"label":p,"value":p} for p in sorted(df_operational["Sorting_Process"].unique())],
                value=sorted(df_operational["Sorting_Process"].unique())[0]
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"1%"}),
        html.Div([
            html.Label("Worker Experience (yrs)"),
            dcc.Dropdown(
                id="rq2-experience",
                options=[{"label":str(e),"value":e} for e in sorted(df_operational["Worker_Experience_Years"].unique())],
                value=int(df_operational["Worker_Experience_Years"].median())
            )
        ], style={"width":"24%","display":"inline-block","marginLeft":"1%"})
    ], style={"padding":"20px"}),
    dcc.Graph(id="rq2-prediction-chart"),

    # === 11) Model Validation ===
    html.H2("🔍 Model Validation Metrics", style={"textAlign":"center"}),
    dcc.Graph(id="model-eval", figure=fig_eval)

])

# ─── 7) (All your existing callbacks go here unchanged) ──────────────────────────
@app.callback(
    Output("yearly-waste-type","figure"),
    Output("monthly-recycle-use","figure"),
    Output("impact-comparison","figure"),
    Input("company-dropdown","value"),
    Input("filter-year","value"),
    Input("filter-month","value"),
    Input("filter-waste","value")
)
def update_company_graphs(selected_company, fy, fm, fw):
    dff = df[df["Company"]==selected_company].copy()
    if fy not in (None,"All"):
        dff = dff[dff["Year"]==int(fy)]
    if fm not in (None,"All"):
        dff = dff[dff["Month"]==fm]
    if fw not in (None,"All"):
        dff = dff[dff["Waste_Type"]==fw]

    fig1 = px.bar(
        dff.groupby(["Year","Waste_Type"])["Waste_Quantity_Kg"].sum().reset_index(),
        x="Year", y="Waste_Quantity_Kg", color="Waste_Type",
        title="Filtered Yearly Waste Quantity by Type"
    )
    fig2 = px.line(
        dff.groupby(["Month","Recycle_Use"])["Waste_Quantity_Kg"].sum().reset_index(),
        x="Month", y="Waste_Quantity_Kg", color="Recycle_Use",
        title="Monthly Recycle Use Trends"
    )
    fig3 = px.scatter(
        dff, x="Year", y="Environmental_Impact_Rating",
        color="Waste_Type", size="Waste_Quantity_Kg",
        hover_data=["Recycle_Use"],
        title="Environmental Impact by Waste Type"
    )
    return fig1, fig2, fig3

@app.callback(
    Output("monthly-recycle-use-filtered","figure"),
    Input("year-filter","value"),
    Input("month-filter","value"),
    Input("company-filter","value")
)
def update_filtered_monthly_recycle(sy, sm, sc):
    dff = df.copy()
    if sy not in (None,"All"):
        dff = dff[dff["Year"]==int(sy)]
    if sm not in (None,"All"):
        dff = dff[dff["Month"]==sm]
    if sc not in (None,"All"):
        dff = dff[dff["Company"]==sc]

    grouped = dff.groupby(["Month","Recycle_Use"])["Waste_Quantity_Kg"].sum().reset_index()
    return px.bar(
        grouped, x="Month", y="Waste_Quantity_Kg", color="Recycle_Use",
        title="Filtered Monthly Recycle Use Trends"
    )

@app.callback(
    Output("impact-custom-graph","figure"),
    Input("impact-company","value"),
    Input("impact-year","value"),
    Input("impact-month","value"),
    Input("impact-waste","value")
)
def update_impact_graph(co, ye, mo, wt):
    dff = df.copy()
    if co not in (None,"All"):
        dff = dff[dff["Company"]==co]
    if ye not in (None,"All"):
        dff = dff[dff["Year"]==int(ye)]
    if mo not in (None,"All"):
        dff = dff[dff["Month"]==mo]
    if wt not in (None,"All"):
        dff = dff[dff["Waste_Type"]==wt]
    if dff.empty:
        return px.scatter(title="No data found")
    return px.scatter(
        dff, x="Year", y="Environmental_Impact_Rating",
        color="Waste_Type", size="Waste_Quantity_Kg",
        hover_data=["Company","Month","Recycle_Use"],
        title="Environmental Impact by Waste Type"
    )

@app.callback(
    Output("ready-items-output","children"),
    Input("waste-type-single","value"),
    Input("waste-type-multi","value"),
    Input("recycle-type-single","value"),
    Input("recycle-type-multi","value")
)
def show_ready_for_sell(ws, wm, rs, rm):
    dff = df[df["Sell_Status"]=="ready for sell"].copy()
    if ws not in (None,"All"):
        dff = dff[dff["Waste_Type"]==ws]
    elif wm:
        dff = dff[dff["Waste_Type"].isin(wm)]
    if rs not in (None,"All"):
        dff = dff[dff["Recycle_Use"]==rs]
    elif rm:
        dff = dff[dff["Recycle_Use"].isin(rm)]
    if dff.empty:
        return html.P("No items match filters.")
    summary = dff.groupby(["Company","Waste_Type","Recycle_Use"])["Waste_Quantity_Kg"].sum().reset_index()
    return html.Div([
        dcc.Graph(figure=px.bar(
            summary, x="Company", y="Waste_Quantity_Kg",
            color="Waste_Type", hover_data=["Recycle_Use"],
            title="Ready for Sell Inventory"
        ))
    ])

@app.callback(
    Output("company-waste-comparison","figure"),
    Input("waste-comparison-type","value"),
    Input("waste-comparison-year","value")
)
def update_company_waste(wt, yr):
    dff = df.copy()
    if wt not in (None,"All"):
        dff = dff[dff["Waste_Type"]==wt]
    if yr not in (None,"All"):
        dff = dff[dff["Year"]==int(yr)]
    grouped = dff.groupby("Company")["Waste_Quantity_Kg"].sum().reset_index()
    if grouped.empty:
        return px.pie(names=[], values=[], title="No data")
    return px.pie(
        grouped, names="Company", values="Waste_Quantity_Kg",
        hole=0.4, title="Company-wise Waste Contribution"
    )

@app.callback(
    Output("tradeoff-graph","figure"),
    Output("summary-metrics","children"),
    Input("tradeoff-company","value"),
    Input("tradeoff-year","value"),
    Input("tradeoff-month","value")
)
def update_tradeoff_graph(company, year, month):
    dff = df_operational.copy()
    if company not in (None,"All"):
        dff = dff[dff["Company"]==company]
    if year not in (None,"All"):
        dff = dff[dff["Year"]==int(year)]
    if month not in (None,"All"):
        dff = dff[dff["Month"]==month]

    grouped = (
        dff.groupby("Recycle_Use")
           .agg({
               "Waste_Quantity_Kg":"sum",
               "Transaction_Value_USD":"sum",
               "Environmental_Impact_Rating":"mean"
           })
           .reset_index()
    )
    if grouped.empty:
        return px.scatter(title="No data"), html.P("No data to display.")
    fig = px.scatter(
        grouped,
        x="Transaction_Value_USD", y="Waste_Quantity_Kg",
        size="Environmental_Impact_Rating", color="Recycle_Use",
        hover_data=["Environmental_Impact_Rating"],
        title="Trade-Offs: Revenue vs Waste vs Impact"
    )
    metrics = html.Ul([
        html.Li(f"Recycle Use Types: {grouped['Recycle_Use'].nunique()}"),
        html.Li(f"Total Waste: {grouped['Waste_Quantity_Kg'].sum():,.2f} kg"),
        html.Li(f"Total Revenue: ${grouped['Transaction_Value_USD'].sum():,.2f}"),
        html.Li(f"Avg Impact Rating: {grouped['Environmental_Impact_Rating'].mean():.2f}")
    ])
    return fig, metrics

@app.callback(
    Output("yearly-waste-fig","figure"),
    Output("yearly-co2-fig","figure"),
    Input("perf-company","value"),
    Input("perf-year","value"),
    Input("perf-month","value")
)
def update_performance(company, year, month):
    dff = df_operational.copy()
    if company not in (None,"All"):
        dff = dff[dff["Company"]==company]
    if year not in (None,"All"):
        dff = dff[dff["Year"]==int(year)]
    if month not in (None,"All"):
        dff = dff[dff["Month"]==month]

    waste_year = dff.groupby(["Year","Waste_Type"])["Waste_Quantity_Kg"].sum().reset_index()
    fig_w = px.bar(
        waste_year, x="Year", y="Waste_Quantity_Kg",
        color="Waste_Type", title="Yearly Waste by Type"
    )
    co2_year = dff.groupby(["Year","Waste_Type"])["CO2_kg"].sum().reset_index()
    fig_c = px.bar(
        co2_year, x="Year", y="CO2_kg",
        color="Waste_Type", title="Yearly CO₂-eq by Type",
        labels={"CO2_kg":"CO₂-eq (kg)"}
    )
    return fig_w, fig_c

@app.callback(
    Output("tradeoff-scatter","figure"),
    Output("tradeoff-summary","children"),
    Input("trade-company","value"),
    Input("trade-year","value"),
    Input("trade-month","value")
)
def update_tradeoff_co2(company, year, month):
    dff = df_operational.copy()
    if company not in (None,"All"):
        dff = dff[dff["Company"]==company]
    if year not in (None,"All"):
        dff = dff[dff["Year"]==int(year)]
    if month not in (None,"All"):
        dff = dff[dff["Month"]==month]

    grp = dff.groupby("Recycle_Use").agg({
        "Waste_Quantity_Kg":"sum",
        "Transaction_Value_USD":"sum",
        "CO2_kg":"sum"
    }).reset_index()

    fig = px.scatter(
        grp, x="Transaction_Value_USD", y="Waste_Quantity_Kg",
        size="CO2_kg", color="Recycle_Use",
        hover_data=["CO2_kg"],
        title="Trade-Off: Revenue vs Waste vs CO₂-eq",
        labels={"CO2_kg":"CO₂-eq (kg)"}
    )
    summary = html.Ul([
        html.Li(f"Types: {grp['Recycle_Use'].nunique()}"),
        html.Li(f"Total Waste: {grp['Waste_Quantity_Kg'].sum():,.2f} kg"),
        html.Li(f"Total Revenue: ${grp['Transaction_Value_USD'].sum():,.2f}"),
        html.Li(f"Total CO₂-eq: {grp['CO2_kg'].sum():,.2f} kg")
    ])
    return fig, summary

@app.callback(
    Output("forecast-waste-compare","figure"),
    Output("forecast-recycle-compare","figure"),
    Output("forecast-co2-compare","figure"),
    Input("pred-company","value"),
    Input("pred-month","value"),
    Input("pred-year","value")
)
def update_forecasts(company, month, year):
    dff = df_operational.copy()
    if company not in (None,"All"):
        dff = dff[dff["Company"]==company]
    if month not in (None,"All"):
        dff = dff[dff["Month"]==month]

    models = {
        "Linear": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=0),
        "GradientBoosting": GradientBoostingRegressor(random_state=0)
    }

    # Waste forecasts
    waste_hist = dff.groupby(["Year","Waste_Type"])["Waste_Quantity_Kg"].sum().reset_index()
    recs = []
    for name, mdl in models.items():
        for wt in waste_hist["Waste_Type"].unique():
            sub = waste_hist[waste_hist["Waste_Type"]==wt]
            if len(sub)>=2:
                X,y = sub["Year"].values.reshape(-1,1), sub["Waste_Quantity_Kg"].values
                mdl.fit(X,y)
                preds = mdl.predict([[yr] for yr in future_years])
                for yr,p in zip(future_years,preds):
                    recs.append({"Year":yr,"Waste_Type":wt,
                                 "Waste_Quantity_Kg":max(p,0),"Model":name})
    dfw = pd.DataFrame(recs)
    if year not in (None,"All"):
        dfw = dfw[dfw["Year"]==int(year)]
    fig_wf = px.bar(
        dfw, x="Year", y="Waste_Quantity_Kg", color="Model",
        facet_col="Waste_Type", facet_col_wrap=3, barmode="group",
        title="Forecast: Waste by Type"
    )

    # Recycle forecasts
    rec_hist = dff.groupby(["Year","Recycle_Use"])["Waste_Quantity_Kg"].sum().reset_index()
    recs=[]
    for name, mdl in models.items():
        for ru in rec_hist["Recycle_Use"].unique():
            sub = rec_hist[rec_hist["Recycle_Use"]==ru]
            if len(sub)>=2:
                X,y = sub["Year"].values.reshape(-1,1), sub["Waste_Quantity_Kg"].values
                mdl.fit(X,y)
                preds = mdl.predict([[yr] for yr in future_years])
                for yr,p in zip(future_years,preds):
                    recs.append({"Year":yr,"Recycle_Use":ru,
                                 "Waste_Quantity_Kg":max(p,0),"Model":name})
    dfr = pd.DataFrame(recs)
    if year not in (None,"All"):
        dfr = dfr[dfr["Year"]==int(year)]
    fig_rf = px.bar(
        dfr, x="Year", y="Waste_Quantity_Kg", color="Model",
        facet_col="Recycle_Use", facet_col_wrap=3, barmode="group",
        title="Forecast: Recycle by Type"
    )

    # CO₂ forecasts
    co2_hist = dff.groupby("Year")["CO2_kg"].sum().reset_index()
    recs=[]
    for name, mdl in models.items():
        if len(co2_hist)>=2:
            X,y = co2_hist["Year"].values.reshape(-1,1), co2_hist["CO2_kg"].values
            mdl.fit(X,y)
            for yr in future_years:
                recs.append({"Year":yr,"CO2_kg":mdl.predict([[yr]])[0],"Model":name})
    dfc = pd.DataFrame(recs)
    if year not in (None,"All"):
        dfc = dfc[dfc["Year"]==int(year)]
    fig_cf = px.line(
        dfc, x="Year", y="CO2_kg", color="Model",
        title="Forecast: Total CO₂-eq Impact",
        labels={"CO2_kg":"CO₂-eq (kg)"}
    )

    return fig_wf, fig_rf, fig_cf

@app.callback(
    Output("rq2-prediction-chart","figure"),
    Input("rq2-company","value"),
    Input("rq2-cutting","value"),
    Input("rq2-sorting","value"),
    Input("rq2-experience","value")
)
def update_whatif(company, cutting, sorting, experience):
    row = pd.DataFrame([{
        "Company":company,
        "Cutting_Technique":cutting,
        "Sorting_Process":sorting,
        "Worker_Experience_Years":experience
    }])
    X_row = pd.get_dummies(row).reindex(columns=X_rq2.columns, fill_value=0)
    preds = {name:mdl.predict(X_row)[0] for name,mdl in models_rq2.items()}
    dfp = pd.DataFrame({
        "Model":list(preds.keys()),
        "Predicted_Waste_kg":list(preds.values())
    })
    fig = px.line(
        dfp, x="Model", y="Predicted_Waste_kg",
        markers=True,
        title="What-If Predicted Waste by Model",
        labels={"Predicted_Waste_kg":"Waste Quantity (kg)"}
    )
    return fig

if __name__ == "__main__":
    app.run_server(debug=True)
