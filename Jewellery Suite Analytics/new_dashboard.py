import dash
import dash_table
from dash.dependencies import Input, Output
import dash_html_components as html
import pandas as pd
import plotly.express as px
import dash_core_components as dcc
from sklearn.linear_model           import LogisticRegression
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
sales = pd.read_csv('/home/ammy/Downloads/Book1.csv')
sales.dropna()
#RFE
url = "/home/ammy/Downloads/Book1.csv"
dataframe = pd.read_csv(url)
dataframe = dataframe.drop(['Product_ID','Category_ID','Channel_Key','Location_Key','Employee_Key','Customer_Key','Prom_Type_ID'
                     ,'Store_Key','Month','Growth%','DateKey','Year','Sales_Target','Sales_Tartget_Attained','Profit','SalesGrowth'], axis=1)
dataframe1 = dataframe.drop(['Sales_amount'],axis=1)
array = dataframe.values
X = array[:,0:6]
Y = array[:,6]
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)
model = LogisticRegression(solver='lbfgs')
rfe = RFE(model, 3)
fit = rfe.fit(X, training_scores_encoded)

#ust
url = "/home/ammy/Downloads/Book1.csv"
data = pd.read_csv(url)
data = data.drop(['Product_ID','Category_ID','Channel_Key','Location_Key','Employee_Key','Customer_Key','Prom_Type_ID'
                     ,'Store_Key','Month','Growth%','DateKey','Year','Sales_Target','Sales_Tartget_Attained','Profit','SalesGrowth'], axis=1)
X1=data.iloc[:,0:6]  #independent columns
y1=data.iloc[:,-1]  #target column

#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=2)
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(y1)
fit1 = bestfeatures.fit(X1,training_scores_encoded)
dfscores = pd.DataFrame(fit1.scores_)
dfcolumns = pd.DataFrame(X1.columns)
#concat two df for better visualization
featureScore = pd.concat([dfcolumns, dfscores],axis=1)
featureScore.columns = ['Specification','Score']

labelencoder = LabelEncoder()
gold = pd.read_csv('/home/ammy/Downloads/Book1.csv')
gold.iloc[:, 16:17] = labelencoder.fit_transform(gold.iloc[:, 16:17])
gold.iloc[:, 0:1] = labelencoder.fit_transform(gold.iloc[:, 0:1])
gold.iloc[:, 1:2] = labelencoder.fit_transform(gold.iloc[:, 1:2])
gold.iloc[:, 2:3] = labelencoder.fit_transform(gold.iloc[:, 2:3])
gold.iloc[:, 3:4] = labelencoder.fit_transform(gold.iloc[:, 3:4])
gold.iloc[:, 17:18] = labelencoder.fit_transform(gold.iloc[:, 17:18])
#extra tree classifier to show importance of features
model = ExtraTreesClassifier()
X = gold.iloc[:, [0,1,2,3,7, 8,9,10,11,12,16,17]] #independent columns
y = gold.iloc[:,13:14]
y=y.astype('int')
model.fit(X,y)

print(model.feature_importances_)
#default scatter plot
def_scatter =px.scatter(
        x=sales.Sales_amount,
        y=sales.Discount,
        labels={"x": "Sales Amount", "y": "Discount"},
        title=f"comparison ",
    )
#corr
corrmat =gold.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10,10))
#plot heat map
fig23= px.imshow(corrmat)

#extra tree classifier
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
data12 = feat_importances.nlargest(12).plot(kind='barh')
#fig = go.Figure(data=[go.Scatter(x=feat_importances,text=X.columns)])
fig1 = px.bar(feat_importances,labels={'hello','world'},color_discrete_sequence =['orange'])

import base64
image_filename = '/home/ammy/Downloads/prediction.png' # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())
#Layout Section

app.layout = html.Div([
    html.H3('Jewellery Suite Analytics',style={'color':'darkorange','text-align':'center'}),
    html.Div([
        html.Div([
            html.H4('Extratree Classifier',style={'text-align':'center'}),
            dcc.Graph(
                id='example-graph-2',
                figure=fig1, style={'height': '470px'}
            ),
        ], className="six columns"),

        html.Div([
html.H4('Univariate Statistical Test',style={'text-align':'center'}),
            html.Br(),
            html.Br(),
            html.Br(),
            dash_table.DataTable(
                id='table',
                columns=[{"name": i, "id": i} for i in featureScore.columns[0:2]],
                data=featureScore.to_dict('records'),
                style_cell={'padding': '5px',
                            'text-align': 'center',
                            'fontSize': '15px',
                            'backgroundColor': 'rgb(50, 50, 50)',
                            'fontWeight': 'bold',
                            'color': 'white',
                            },
                style_header={
                    'backgroundColor': 'rgb(30, 30, 30)',
                    'fontSize': '20px',
                },
            ),
        ], className="six columns"),
    ], className="row"),

#scatter plots
html.H4('Scatter Plot',style={'text-align':'center'}),
    html.H3('Sales Amount Vs ',style={'color': 'aqua'}),
    html.Button('Discount', id='btn-nclicks-1', n_clicks=0),
    html.Button('Making Charges+TAX', id='btn-nclicks-2', n_clicks=0),
    html.Button('Actual Cost Price', id='btn-nclicks-3', n_clicks=0),
    html.Button('Quantity Ounce', id='btn-nclicks-4', n_clicks=0),
    html.Button('Special Discount', id='btn-nclicks-5', n_clicks=0),
    html.Button('Product ', id='btn-nclicks-6', n_clicks=0),
    html.Button('Prom Type', id='btn-nclicks-7', n_clicks=0),
    html.Button('Category ', id='btn-nclicks-8', n_clicks=0),
    html.Div(dcc.Graph(id='container-button-timestamp',figure=def_scatter)),

    #correlation heatmap
html.H4('Features Scatterplot',style={'text-align':'center'}),
dcc.Graph(
                id='example-graph-3',
                figure=fig23, style={'height': '650px','width':'1300px'}
            ),

    #Predicting using deep learning algorithm
html.H4('Prediction',style={'text-align':'center'}),
html.Img(src='data:/home/ammy/Downloads/prediction.png;base64,{}'.format(encoded_image.decode()),
                style={'height': '500px','width':'1000px','margin-left':'190px'})
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