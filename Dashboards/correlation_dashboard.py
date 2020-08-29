import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
#import plotly.graph_objs as go
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
sales = pd.read_csv('/home/ammy/Downloads/Book1.csv')
sales.dropna()
app.layout = html.Div([
    html.H2('Correlation Dashboard',style={'color': 'orange','text-align': 'center'}),
    html.H4('Sales Amount Vs ',style={'color': 'blue'}),
    html.Hr(style={'color': 'blue','height': '5px',
            'background': 'black' }),

    html.Button('Discount', id='btn-nclicks-1', n_clicks=0),
    html.Button('Making Charges+TAX', id='btn-nclicks-2', n_clicks=0),
    html.Button('Actual Cost Price', id='btn-nclicks-3', n_clicks=0),
    html.Button('Quantity Ounce', id='btn-nclicks-4', n_clicks=0),
    html.Button('Special Discount', id='btn-nclicks-5', n_clicks=0),
    html.Button('Product ', id='btn-nclicks-6', n_clicks=0),
    html.Button('Prom Type', id='btn-nclicks-7', n_clicks=0),
    html.Button('Category ', id='btn-nclicks-8', n_clicks=0),
    html.Div(dcc.Graph(id='container-button-timestamp',figure={}))
])
@app.callback(Output('container-button-timestamp', "figure"),
              [Input('btn-nclicks-1', 'n_clicks'),
               Input('btn-nclicks-2', 'n_clicks'),
               Input('btn-nclicks-3', 'n_clicks'),
               Input('btn-nclicks-4', 'n_clicks'),
               Input('btn-nclicks-5', 'n_clicks'),
               Input('btn-nclicks-6', 'n_clicks'),
               Input('btn-nclicks-7', 'n_clicks'),
               Input('btn-nclicks-8', 'n_clicks')])

def displayClick(btn1, btn2, btn3,btn4,btn5,btn6,btn7,btn8):
    global msg
    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'btn-nclicks-1' in changed_id:
        msg =px.scatter(
        x=sales.Sales_amount,
        y=sales.Discount,
        labels={"x": "Sales Amount", "y": "Discount"},
        title=f"comparison ",
    )
    elif 'btn-nclicks-2' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Making_charges_TAX,
            labels={"x": "Sales amt", "y": "Making_charges_TAX"},
            title=f"comparison ",
        )
    elif 'btn-nclicks-3' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Actual_Cost_Price,
            labels={"x": "Sales amount", "y": "Actual Cost Price"},
            title=f"comparison ",
        )
    elif 'btn-nclicks-4' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Quantity_Ounce,
            labels={"x": "Sales amount", "y": "Quantity Ounce"},
            title=f"comparison ",
        )
    elif 'btn-nclicks-5' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Special_Discount,
            labels={"x": "Sales amount", "y": "Special Discount"},
            title=f"comparison ",
        )
    elif 'btn-nclicks-6' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Product_ID,
            labels={"x": "Sales amount", "y": "Product ID"},
            title=f"comparison ",
        )
    elif 'btn-nclicks-7' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Prom_Type_ID,
            labels={"x": "Sales amount", "y": "Prom Type ID"},
            title=f"comparison ",
        )
    elif 'btn-nclicks-8' in changed_id:
        msg = px.scatter(
            x=sales.Sales_amount,
            y=sales.Category_ID,
            labels={"x": "Sales amount", "y": "Category ID"},
            title=f"comparison ",
        )
    msg.update_layout(transition_duration=500)
    return msg

if __name__ == '__main__':
    app.run_server(debug=True,dev_tools_ui=False,dev_tools_props_check=False)
