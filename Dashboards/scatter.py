import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)


df = pd.read_csv("/home/ammy/Downloads/intro_bees.csv")
sales = pd.read_csv('/home/ammy/Downloads/Book1.csv')
df = df.groupby(['State', 'ANSI', 'Affected by', 'Year', 'state_code'])[['Pct of Colonies Impacted']].mean()
df.reset_index(inplace=True)
print(df[:5])

np.random.seed(50)
x_rand = np.random.randint(1,61,60)
y_rand = np.random.randint(1,61,60)


app.layout = html.Div([

    html.H1("Web Application Dashboards", style={'text-align': 'center'}),

    dcc.Dropdown(id="slct_year",
                 options=[
                     {"label": "2015", "value": 2015},
                     {"label": "2016", "value": 2016},
                     {"label": "2017", "value": 2017},
                     {"label": "2018", "value": 2018}],
                 multi=False,
                 value=2015,
                 style={'width': "40%"}
                 ),

    html.Div(id='output_container', children=[]),
    html.Br(),

    dcc.Graph(id='my_bee_map', figure={}),
dcc.Graph(id='avp-graph', figure={}),
dcc.Graph(id='avp-graph1', figure={}),
dcc.Graph(id='avp-graph2', figure={}),
dcc.Graph(id='avp-graph3', figure={}),
    dcc.Graph(id='avp-graph4', figure={}),






])




@app.callback(
    [Output(component_id='output_container', component_property='children'),
     Output(component_id='my_bee_map', component_property='figure'),
     Output(component_id="avp-graph", component_property="figure"),
     Output(component_id="avp-graph1", component_property="figure"),
     Output(component_id="avp-graph2", component_property="figure"),
     Output(component_id="avp-graph3", component_property="figure"),
     Output(component_id="avp-graph4", component_property="figure")],
    [Input(component_id='slct_year', component_property='value')]
)
def update_graph(option_slctd):
    print(option_slctd)
    print(type(option_slctd))

    container = "The year chosen by user was: {}".format(option_slctd)

    dff = df.copy()
    dff = dff[dff["Year"] == option_slctd]
    dff = dff[dff["Affected by"] == "Varroa_mites"]

    fig = px.choropleth(
        data_frame=dff,
        locationmode='USA-states',
        locations='state_code',
        scope="usa",
        color='Pct of Colonies Impacted',
        hover_data=['State', 'Pct of Colonies Impacted'],
        color_continuous_scale=px.colors.sequential.YlOrRd,
        labels={'Pct of Colonies Impacted': '% of Bee Colonies'},
        template='plotly_dark'
    )

    avp_fig = px.scatter(
        x=sales.Sales_amount,
        y=sales.Sales_Unit_Price,
        labels={"x": "Sales Amt", "y": "Sales unit price"},
        title=f"comparison ",
    )
    avp_fig1 = px.scatter(
        x=sales.Sales_amount,
        y=sales.Discount,
        labels={"x": "Sales Amount", "y": "Discount"},
        title=f"comparison ",
    )
    avp_fig2 = px.scatter(
        x=sales.Sales_amount,
        y=sales.Special_Discount,
        labels={"x": "Sales Amount", "y": "Special Dicount"},
        title=f"comparison ",
    )

    avp_fig3 = px.scatter(
        x=sales.Sales_amount,
        y=sales.Making_charges_TAX,
        labels={"x": "SAles amt", "y": "Making_charges_TAX"},
        title=f"comparison ",
    )

    avp_fig4 = px.scatter(
        x=sales.Sales_amount,
        y=sales.Actual_Cost_Price,
        labels={"x": "Sales amount", "y": "Actual Cost Price"},
        title=f"comparison ",
    )

    return container, fig,avp_fig, avp_fig1, avp_fig2,avp_fig3,avp_fig4



if __name__ == '__main__':
    app.run_server(debug=True)