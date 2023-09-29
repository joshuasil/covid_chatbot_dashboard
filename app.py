import pandas as pd
import numpy as np
import statistics as stat
from get_postgres_str import get_postgres_str
## Plotly and Dash Imports
import dash
import dash_bootstrap_components as dbc
from dash import dcc, Dash
from dash import html, callback_context
from dash.dependencies import Input, Output, State, ClientsideFunction
import plotly.express as px
from dash.exceptions import PreventUpdate
from PIL import Image

## SQL Imports
from flask import Flask
import psycopg2
from sqlalchemy import create_engine, text
from dash_bootstrap_templates import ThemeChangerAIO, template_from_url


# PostgreSQL

## Postgres username, password, and database name
postgres_str = get_postgres_str()

## Create the Connection
engine = create_engine(postgres_str, echo=False)
conn = engine.connect()

## Creating our Dataframe
sql_select_query = text('''SELECT * FROM public.covid_logs;''')

sqlresult = conn.execute(sql_select_query)
df_comp = pd.DataFrame(sqlresult.fetchall())
df_comp.columns = sqlresult.keys()
df_comp['request_timestamp'] = df_comp['request_timestamp'].fillna('2021-03-24 21:49:42.601766') # For this, we are just filling the 
# first dates, for which we have no time, with the first actual time we have
df_comp = df_comp.sort_values('request_timestamp') # Making sure everything is in order by when the bot received the query

# Creating our various descriptors we can filter by

## Website
df_comp['websites'] = df_comp['page_url'].apply(lambda x: None if (x is None) else x.replace('https://','').split('/')[0])
websites = df_comp['websites'].apply(lambda x: None if (x is None) else x.replace('https://','').split('/')[0]).value_counts().reset_index().loc[:9,'websites'].to_list()
df_comp.loc[~df_comp['websites'].isin(websites),'websites'] = 'others'
websites = df_comp['websites'].unique()

## Platform
df_comp['platform'] = df_comp['type_bot'].replace({'colorado_covid_':'','dch_VW_':'','dch_stride_':''}, regex=True)
platform = list(df_comp['platform'].unique())

## Type of Chatbot
df_comp['chatbots'] = df_comp['type_bot'].replace({'_web':'','_text':'','_twilio/incognito_browser':''}, regex=True)
chatbots = list(df_comp['chatbots'].unique())


# Descriptive Statistics

df_user_statistics = df_comp['conversation_id'].value_counts().rename_axis('users').reset_index(name='counts')
unique_users = df_comp['conversation_id'].nunique()
tot_questions = df_comp.shape[0]
avg_mess_per_user = round(df_user_statistics['counts'].mean(),2)
minimum_mess_per_user = df_user_statistics['counts'].min()
maximum_mess_per_user = df_user_statistics['counts'].max()


# Performance Statistics

## Getting our dataframe of dates as indices and the accuracy, precision, and picklist percent as a cumulative up to that day
df_count_by_date = df_comp['request_timestamp'].dt.date.value_counts().sort_index().rename_axis('dates').reset_index(name='counts')
dates = [str(x) for x in list(df_count_by_date['dates'])] # Getting the sorted, unique dates
df_comp['yyyymmdd'] = [str(x) for x in df_comp['request_timestamp'].dt.date] # Casting the dates (multiples) as strings to be
# compared to the string of unique dates
### Dataframe skeleton
df_perf_stats = np.zeros((len(dates), 3))
df_perf_stats = pd.DataFrame(df_perf_stats, columns = ['accuracy', 'precision', 'pick_list_percent'])
df_perf_stats.index = dates
### Filling out the dataframe
for dat in dates:
    date_idx = [idx for idx, value in enumerate(list(df_comp['yyyymmdd'])) if value == dat][-1]
    correct_labels_for_df = df_comp['correct_intent'].iloc[:date_idx]
    
    n_yes = len([x for x in correct_labels_for_df if x == 'Yes'])
    n_no = len([x for x in correct_labels_for_df if x == 'No'])
    n_pick_list_yes = len([x for x in correct_labels_for_df if x == 'Pick List Yes'])
    n_pick_list_no = len([x for x in correct_labels_for_df if x == 'Pick List No'])
    accuracy_df = round(100*(n_yes/(n_yes+n_pick_list_no+n_no)),2)
    precision_df = round(100*(n_yes+n_pick_list_yes)/(n_yes+n_pick_list_yes+n_pick_list_no+n_no),2)
    percentage_picked_from_pick_list_df = round(100*n_pick_list_yes/(n_pick_list_yes+n_pick_list_no), 2)

    df_perf_stats.at[dat, 'accuracy'] = accuracy_df
    df_perf_stats.at[dat, 'precision'] = precision_df
    df_perf_stats.at[dat, 'pick_list_percent'] = percentage_picked_from_pick_list_df

## Cumulative Accuracy, precision, and pick list percent 

accuracy = df_perf_stats.loc[df_perf_stats.index[-1], 'accuracy']
precision = df_perf_stats.loc[df_perf_stats.index[-1], 'precision']
percentage_picked_from_pick_list = df_perf_stats.loc[df_perf_stats.index[-1], 'pick_list_percent']


# Aesthetics

colors = {
    'background': '#FEFFFF',
    'graph background': '#CDD0D0',
    'title text': '#012337',
    'intent': 'oranges', # Will use continuous color sequence
    'source': '#EF8B69', # Will use discrete color sequence
    'browser': '#F1E091', # Will use discrete color sequence
    'hour': 'greens', # Will use continuous color sequence
    'subtitle text': '#012337',
    'label text': '#012337',
    'line color': '#056B7D'
}

covid_logo = Image.open('COVID_chatbot_logo.png')
sph_logo = Image.open('coloradosph_stacked_schools.jpg')


# Setting up the Figures


## Performance by Date
fig_perf = px.line(df_perf_stats, x = df_perf_stats.index, y=['accuracy', 'precision', 'pick_list_percent'], title = 'Cumulative Accuracy, Precision, and Pick List Activity by Day',
labels = {'index': 'Date', 'value':'Percentage'}, color_discrete_sequence=['#56D9EA', '#7856EA', '#C556EA'], render_mode='webg1')
newnames_perf = {'accuracy': 'Accuracy', 'precision': 'Precision', 'pick_list_percent': 'Pick List Percent'}
fig_perf.for_each_trace(lambda t: t.update(name = newnames_perf[t.name]))
fig_perf.update_layout(title_x=0.5)
fig_perf.update_xaxes(rangeslider_visible=True)

## Count and Cumulative Sum by Date
df_count_by_date = df_comp['request_timestamp'].dt.date.value_counts().sort_index().rename_axis('dates').reset_index(name='counts')
fig_count_by_date = px.line(df_count_by_date, x='dates', y='counts', title = 'Count by Day',
labels = {'dates': 'Date', 'counts': 'Count'}, color_discrete_sequence=[colors['line color']], render_mode='webg1')
fig_count_by_date.update_layout(title_x=0.5)
fig_count_by_date.update_xaxes(rangeslider_visible=True)

df_count_by_date['cum_sum'] = df_count_by_date['counts'].cumsum()
fig_cum_sum_by_date = px.line(df_count_by_date, x='dates', y=['cum_sum', 'counts'], title = 'Cumulative Count by Day',
labels = {'dates': 'Date', 'cum_sum': 'Cumulative Sum', 'counts':'Counts'}, color_discrete_sequence=[colors['line color']], render_mode='webg1')
fig_cum_sum_by_date.update_layout(title_x=0.5)
fig_cum_sum_by_date.update_xaxes(rangeslider_visible=True)

## Count by Intent
n_intents = [5, 10, 15]
count_by_intent = df_comp[df_comp['intent']!='Not Question']
count_by_intent = count_by_intent[count_by_intent['intent']!='Greeting']
count_by_intent = count_by_intent[count_by_intent['intent']!='Switch Languages']
count_by_intent = count_by_intent[count_by_intent['intent']!='Numerical']['intent'].value_counts().rename_axis('intent').reset_index(name='counts')[:5]
fig_intent = px.bar(count_by_intent, y='intent', x="counts", orientation='h', title = 'Top Question Types', color = 'counts',
labels = {'intent': 'Intent', 'counts': 'Count'}, color_continuous_scale = colors['intent'])
fig_intent.update_layout(title_x=0.5)

## Count by Source
count_by_website = df_comp['websites'].value_counts().rename_axis('website').reset_index(name='counts')
fig_website = px.pie(count_by_website, values='counts', names='website', title='Website/Source Percentages',
labels = {'counts': 'Count', 'websites': 'Website/Source'}, color_discrete_sequence=[colors['source']])
fig_website.update_layout(title_x=0.5)

## Count by Browser
df_comp['browser_os_context'].fillna('unknown',inplace=True)
browser_os = df_comp['browser_os_context'].unique()
count_by_browser = df_comp['browser_os_context'].value_counts(normalize=True).rename_axis('browser').reset_index(name='counts')
fig_browser =px.pie(count_by_browser, values='counts', names='browser', title='Browser Percentages',
labels = {'counts': 'count', 'hour': 'Hour'}, color_discrete_sequence=[colors['browser']])
fig_browser.update_layout(title_x=0.5)

## Count by Hour
count_by_hour = df_comp['request_timestamp'].dt.hour.value_counts().sort_index().rename_axis('hour').reset_index(name='counts')
fig_hour = px.bar(count_by_hour, y="counts", x="hour", title = 'Count by Hour', color = 'hour',
labels = {'counts': 'Count', 'hour': 'Hour'}, color_continuous_scale = colors['hour'])
fig_hour.update_layout(title_x=0.5)


cards_global = [
    dbc.Card(
        [
            html.P("Total Unique Users"),
            html.H6(unique_users,id='unique_users'),
        ],
        body=True,
        color="primary",
        inverse=True,
        style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
    ),
    dbc.Card(
        [
            html.P("Total Questions"),
            html.H6(tot_questions,id='tot_questions'),
        ],
        body=True,
        color="primary",
        inverse=True,
        style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
    ),
    dbc.Card(
                [
                    html.P("Avg No. of Messages"),
                    html.H6(avg_mess_per_user,id='avg_mess_per_user'),
                ],
                body=True,
                color="primary",
                inverse=True,
                style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
            ),
    dbc.Card(
                [
                    html.P("Min No. of Messages"),
                    html.H6(minimum_mess_per_user,id='minimum_mess_per_user'),
                ],
                body=True,
                color="primary",
                inverse=True,
                style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
            ),
    dbc.Card(
                [
                    html.P("Max No. of Messages"),
                    html.H6(maximum_mess_per_user,id='maximum_mess_per_user'),
                ],
                body=True,
                color="primary",
                inverse=True,
                style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
            )
]

cards_perf = [
    dbc.Card(
        [
            # html.P("Accuracy",className="card-title"),
            # html.H6(accuracy,className="card-text", id='accuracy'),
            html.P("Accuracy"),
            html.H6(accuracy,id='accuracy'),
        ],
        body=True,
        color="primary",
        inverse=True,
        style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
    ),
    dbc.Card(
        [
            html.P("Precision"),
            html.H6(precision,id='precision'),
        ],
        body=True,
        color="primary",
        inverse=True,
        style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
    ),
    dbc.Card(
                [
                    html.P("Pick list %"),
                    html.H6(percentage_picked_from_pick_list,id='percentage_picked_from_pick_list'),
                ],
                body=True,
                color="primary",
                inverse=True,
                style={'textAlign': 'center',"height": 55,"line-height":'0px'},
        className="mb-1"
            )
]



tabs_fig = dcc.Tabs([
        dcc.Tab(label='Count by Date', children=[
            dcc.Graph(id='count_by_date', figure=fig_count_by_date)
        ]),
        dcc.Tab(label='Cum Sum by Date', children=[
            dcc.Graph(id='cum_count_by_date', figure=fig_cum_sum_by_date)
        ]),
        dcc.Tab(label='Performance', children=[
            dcc.Graph(id='performance_by_date', figure=fig_perf)
        ]),
        dcc.Tab(label='Intent', children=[
            dcc.Graph(id='count_by_intent', figure=fig_intent)
        ]),
        dcc.Tab(label='Website', children=[
            dcc.Graph(id='count_by_website', figure=fig_website)
        ]),
        dcc.Tab(label='Hour', children=[
            dcc.Graph(id='count_by_hour', figure=fig_hour)
        ]),
        dcc.Tab(label='Browser', children=[
            dcc.Graph(id='count_by_browser', figure=fig_browser)
        ]),
    ])

navbar = dbc.NavbarSimple(
    children=[html.Img(src=sph_logo,height='40px'),
    ],
    brand="mHealth Chatbot Interactive Dashboard",
    brand_href="#",
    color="primary",
    dark=True,
)


dropdown = dbc.Row(
    [
        dbc.Col(
            dbc.DropdownMenu(children=[
                            html.Button('Select/Unselect all', id='select_all_chatbots', n_clicks=0, className="btn btn-success",style={"margin-left": 10}),
        dcc.Checklist(id='chatbots',
                        options = [{'value':'colorado_covid', 'label':'Colorado COVID'},
                        {'value':'covid_nm', 'label':'New Mexico COVID'}, {'value':'dch_VW', 'label':'Valley Wide DCH'},
                        {'value':'dch_stride', 'label':'Stride DCH'}], 
                        value = chatbots,labelStyle = {'display': 'block'},
                        inputStyle={"margin-right": 10,"margin-bottom": 10,"margin-top": 10},
                        className="form-check"),
        ],
        label="chatbots"
    ),
            #width="auto",
        ),
        dbc.Col(
            dbc.DropdownMenu(children=[
                html.Button('Select/Unselect all', id='select_all_platform', n_clicks=0, className="btn btn-success",style={"margin-left": 10}),
                dcc.Checklist(id='platform',  
                    options = [{'value':'text', 'label':'Text'}, {'value':'web', 'label':'Web'},
                    {'value':'twilio/incognito_browser', 'label':'Twilio/Incognito'}, 
                    {'value':'covid_nm_web', 'label':'New Mexico Web'}],
                    value = platform, labelStyle = {'display': 'block'},
                                            inputStyle={"margin-right": 10,"margin-bottom": 10,"margin-top": 10},
                                            className="form-check")
            ],
                label="platform"
            ),
            #width="auto",
        ),
        dbc.Col(
            dbc.DropdownMenu(label="Website", 
                            children=[
                                html.Button('Select/Unselect all', id='select_all_website', n_clicks=0, className="btn btn-success",style={"margin-left": 10}),
                                dcc.Checklist(id='website_source',
                                        options = [{'label': i, 'value': i} for i in websites], value = websites, labelStyle = {'display': 'block'},
                                            inputStyle={"margin-right": 10,"margin-bottom": 10,"margin-top": 10},
                                            className="form-check", style={"width": "150%"}),
                                    ]),
        ),
        dbc.Col(
            dbc.DropdownMenu(
                label="No. of intents", children=[ dcc.RadioItems(options=[{'label': i, 'value': i} for i in n_intents], 
                                                                    value=5,labelStyle = {'display': 'block'},
                                                                    inputStyle={"margin-right": 10,"margin-bottom": 10,"margin-top": 10},
                                                                    className="form-check",id='n_intents')], 
                direction="down"
            )
        ),
        dbc.Col(
            dbc.DropdownMenu(label='Beginning and End Dates',
                                    children=[dcc.DatePickerRange(
                                            id='begin_date',
                                            min_date_allowed=dates[0],
                                            max_date_allowed=dates[-1],
                                            start_date = dates[0],
                                            end_date = dates[-1]
                                            )],
                        className="form-check")
        ),
        dbc.Col(
            ThemeChangerAIO(aio_id="theme", radio_props={"value":dbc.themes.FLATLY})
        )
    ],
    justify="between",class_name="card-body"
)

navbar_controls = dbc.Navbar(
    dbc.Container(
        [
            dropdown
        ]
    ),
    color="dark",
    dark=True,
)


# The App's Structure

dbc_css = (
    "https://cdn.jsdelivr.net/gh/AnnMarieW/dash-bootstrap-templates@V1.0.1/dbc.min.css"
)
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, dbc_css])

#server = app.server

app.layout = html.Div(style={'padding':10, 'backgroundColor': colors['background']}, children =[html.Div(navbar),html.Div(dropdown,style={'textAlign': 'center',"line-height":'2px'}),
        #dbc.Row(ThemeChangerAIO(aio_id="theme", radio_props={"value":dbc.themes.FLATLY})),
        dbc.Row(
            [dbc.Col(
                    width=2,
                    children=[dbc.Card(
                        [dbc.CardHeader("Cumulative User Statistics"),dbc.CardBody(cards_global)]
                    ),
                    dbc.Card(
                        [dbc.CardHeader("Chatbot Performance"),dbc.CardBody(cards_perf)]
                    )]),
            dbc.Col(
                    width=10,
                    children=[dbc.Row([dbc.Col(tabs_fig)],style={'textAlign': 'center'})]
                                    
                    )

            ])
    ])



# App Callback

@app.callback([
    dash.dependencies.Output('unique_users', 'children'),
    dash.dependencies.Output('tot_questions', 'children'),
    dash.dependencies.Output('avg_mess_per_user', 'children'),
    dash.dependencies.Output('minimum_mess_per_user', 'children'),
    dash.dependencies.Output('maximum_mess_per_user', 'children'),
    dash.dependencies.Output('accuracy', 'children'),
    dash.dependencies.Output('precision', 'children'),
    dash.dependencies.Output('percentage_picked_from_pick_list', 'children'),
    dash.dependencies.Output('count_by_date', 'figure'),
    dash.dependencies.Output('cum_count_by_date', 'figure'),
    dash.dependencies.Output('performance_by_date', 'figure'),
    dash.dependencies.Output('count_by_intent', 'figure'),
    dash.dependencies.Output('count_by_website', 'figure'),
    dash.dependencies.Output('count_by_hour', 'figure'),
    dash.dependencies.Output('count_by_browser', 'figure')],
    [dash.dependencies.Input('chatbots', 'value'),
    dash.dependencies.Input('platform', 'value'),
    dash.dependencies.Input('website_source', 'value'),
    dash.dependencies.Input('n_intents', 'value'),
    dash.dependencies.Input('begin_date', 'start_date'),
    dash.dependencies.Input('begin_date', 'end_date')])



# Callback Function

def date_cum_count_media_type(chatbot_value, platform_value, website_value, n_intents, begin_date, end_date):

    # Filtering
    n_intents = n_intents
    begin_index = [idx for idx, value in enumerate(list(df_comp['yyyymmdd'])) if value == begin_date][0] # We want the first time this date appears
    end_index = [idx for idx, value in enumerate(list(df_comp['yyyymmdd'])) if value == end_date][-1] # We want the last time this date appears
    if begin_index >= end_index: # In case they select the begin after the end
        end_index = begin_index
    if end_index == len(df_comp): # Because python doesn't include the last index in slices, if the last selected index is the 
        # last possible one, we need to just do [begin_index:] to get that last one
        ts = df_comp.iloc[begin_index:]
    else:
        ts = df_comp.iloc[begin_index:end_index]
    ts = ts[ts["chatbots"].isin(chatbot_value)]
    ts = ts[ts["platform"].isin(platform_value)]
    ts = ts[ts["websites"].isin(website_value)] # We now have our dataframe filtered by the selected values

    # Descriptive Statistics
    unique_users = ts['conversation_id'].nunique()
    tot_questions = ts.shape[0]
    df_user_statistics = ts['conversation_id'].value_counts().rename_axis('users').reset_index(name='counts')
    avg_mess_per_user = round(df_user_statistics['counts'].mean(),2)
    minimum_mess_per_user = df_user_statistics['counts'].min()
    maximum_mess_per_user = df_user_statistics['counts'].max()

    # Performance Statistics
    ## Getting the dataframe of accuracy, precision, and pick list percent
    df_dates = ts['request_timestamp'].dt.date.value_counts().sort_index().rename_axis('dates').reset_index(name='counts')
    dates = [str(x) for x in list(df_dates['dates'])] # Getting the sorted, unique dates as strings
    ts['yyyymmdd'] = [str(x) for x in ts['request_timestamp'].dt.date] # Casting the dates (multiples) as strings to be
    # compared to the string of dates
    ### Skeleton of the performance dataframe
    df_perf_stats = np.zeros((len(dates), 3))
    df_perf_stats = pd.DataFrame(df_perf_stats, columns = ['accuracy', 'precision', 'pick_list_percent'])
    df_perf_stats.index = dates
    ### Getting our dataframe of dates as indices and the accuracy, precision, and picklist percent as cumulative up to that day. Note the try/except are to avoid division by zero when selecting options
    for dat in dates:
        date_idx = [idx for idx, value in enumerate(list(ts['yyyymmdd'])) if value == dat][-1] # We include up to the last value of this date
        correct_labels_for_df = ts['correct_intent'][:date_idx]
        
        n_yes = len([x for x in correct_labels_for_df if x == 'Yes'])
        n_no = len([x for x in correct_labels_for_df if x == 'No'])
        n_pick_list_yes = len([x for x in correct_labels_for_df if x == 'Pick List Yes'])
        n_pick_list_no = len([x for x in correct_labels_for_df if x == 'Pick List No'])

        try:
            accuracy_df = round(100*(n_yes/(n_yes+n_pick_list_no+n_no)),2)
        except:
            accuracy_df = 0
        try:
            precision_df = round(100*(n_yes+n_pick_list_yes)/(n_yes+n_pick_list_yes+n_pick_list_no+n_no),2)
        except:
            precision_df = 0
        try:
            percentage_picked_from_pick_list_df = round(100*n_pick_list_yes/(n_pick_list_yes+n_pick_list_no), 2)
        except:
            percentage_picked_from_pick_list_df = 0

        df_perf_stats.at[dat, 'accuracy'] = accuracy_df
        df_perf_stats.at[dat, 'precision'] = precision_df
        df_perf_stats.at[dat, 'pick_list_percent'] = percentage_picked_from_pick_list_df
    ## Cumulative performance statistics. Note the try/except are to avoid division by zero when selecting options
    try: 
        accuracy = df_perf_stats.loc[df_perf_stats.index[-1], 'accuracy']
    except:
        accuracy = 'Accuracy is not applicable.'
    try:
        precision = df_perf_stats.loc[df_perf_stats.index[-1], 'precision']
    except:
        precision = 'Precision is not applicable.'
    try:
        percentage_picked_from_pick_list = df_perf_stats.loc[df_perf_stats.index[-1], 'pick_list_percent']
    except: 
        percentage_picked_from_pick_list = 'Pick list percentage is not applicable.'
        
    # Figures
    ## Count by Date and Hour
    df_count_by_date = ts['request_timestamp'].dt.date.value_counts().sort_index().rename_axis('dates').reset_index(name='counts')
    df_count_by_date['cum_sum'] = df_count_by_date['counts'].cumsum()
    fig_count_by_date = px.line(df_count_by_date, x='dates', y="counts", title = 'Count by Day', labels = {'dates': 'Date', 'counts': 'Count'}, 
                        color_discrete_sequence=[colors['line color']], render_mode='webg1'
                        )
    fig_count_by_date.update_layout(plot_bgcolor=colors['graph background'], paper_bgcolor=colors['background'], font_color=colors['label text'])
    fig_count_by_date.update_layout(title_x=0.5)
    fig_count_by_date.update_xaxes(rangeslider_visible=True)
    fig_cum_sum_by_date = px.line(df_count_by_date, x='dates', y="cum_sum", title = 'Cumulative Count by Day', labels = {'dates': 'Date', 'cum_sum': 'Cumulative Sum'}, color_discrete_sequence=[colors['line color']], render_mode='webg1')
    fig_cum_sum_by_date.update_layout(title_x=0.5)
    fig_cum_sum_by_date.update_layout(plot_bgcolor=colors['graph background'], paper_bgcolor=colors['background'], font_color=colors['label text'])
    fig_cum_sum_by_date.update_xaxes(rangeslider_visible=True)
    ## Performance Figure
    if len(ts) == 0: # If everything in one of the selections is deselected, it causes problems
        fig_perf = px.line(df_perf_stats, x = df_perf_stats.index, title = 'Cumulative Accuracy, Precision, and Pick List Activity by Day',
        labels = {'index': 'Date', 'value':'Percentage'}, color_discrete_sequence=['#F30B58', '#7856EA', '#C556EA'], render_mode='webg1')
        fig_perf.update_layout(plot_bgcolor=colors['graph background'], paper_bgcolor=colors['background'], font_color=colors['label text'])    
        newnames = {'accuracy': 'Accuracy', 'precision': 'Precision', 'pick_list_percent': 'Pick List Percent'}
        fig_perf.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        fig_perf.update_layout(title_x=0.5)
        fig_perf.update_xaxes(rangeslider_visible=True)
    else:
        fig_perf = px.line(df_perf_stats, x = df_perf_stats.index, y=['accuracy', 'precision', 'pick_list_percent'], title = 'Cumulative Accuracy, Precision, and Pick List Activity by Day',
        labels = {'index': 'Date', 'value':'Percentage'}, color_discrete_sequence=['#F30B58', '#7856EA', '#C556EA'], render_mode='webg1')
        fig_perf.update_layout(plot_bgcolor=colors['graph background'], paper_bgcolor=colors['background'], font_color=colors['label text'])    
        newnames = {'accuracy': 'Accuracy', 'precision': 'Precision', 'pick_list_percent': 'Pick List Percent'}
        fig_perf.for_each_trace(lambda t: t.update(name = newnames[t.name]))
        fig_perf.update_layout(title_x=0.5)
        fig_perf.update_xaxes(rangeslider_visible=True)
    ## Count by Intent
    count_by_intent = ts[ts['intent']!='Not Question']
    count_by_intent = count_by_intent[count_by_intent['intent']!='Greeting']
    count_by_intent = count_by_intent[count_by_intent['intent']!='Switch Languages']
    count_by_intent = count_by_intent[count_by_intent['intent']!='Numerical']['intent'].value_counts().rename_axis('intent').reset_index(name='counts')[:n_intents]
    fig_intent = px.bar(count_by_intent, y="intent", x="counts", orientation='h', title = 'Top Question Types', color = 'counts', labels = {'intent': 'Intents', 'counts': 'Count'}, color_continuous_scale = colors['intent'])
    fig_intent.update_layout(plot_bgcolor=colors['graph background'], paper_bgcolor=colors['background'], font_color=colors['label text'])
    fig_intent.update_layout(title_x=0.5)
    ## Count by Hour
    count_by_hour = ts['request_timestamp'].dt.hour.value_counts().sort_index().rename_axis('hour').reset_index(name='counts')
    fig_hour = px.bar(count_by_hour, y="counts", x="hour", title = 'Count by Hour', 
                        color = 'hour', 
                        labels = {'hour': 'Hour', 'counts': 'Count'}, 
                        color_continuous_scale = colors['hour']
                        )
    # fig_hour.update_layout(plot_bgcolor=colors['graph background'], paper_bgcolor=colors['background'], font_color=colors['label text'])
    # fig_hour.update_layout(title_x=0.5)
    ## Count by Browser
    count_by_browser = ts['browser_os_context'].value_counts(normalize=True).rename_axis('browser').reset_index(name='counts')
    fig_browser = px.pie(count_by_browser, values='counts', names='browser', title='Browser Percentages', labels = {'browser': 'Browser', 'counts': 'Count'}, color_discrete_sequence=[colors['browser']])
    fig_browser.update_layout(paper_bgcolor=colors['background'], font_color=colors['label text'])
    fig_browser.update_layout(title_x=0.5)
    ## Count by Source
    count_by_website = ts['websites'].value_counts().rename_axis('website').reset_index(name='counts')
    fig_website = px.pie(count_by_website, values='counts', names='website', title='Website/Source Percentages', labels = {'website': 'Website/Source', 'counts': 'Count'}, color_discrete_sequence=[colors['source']])
    fig_website.update_layout(paper_bgcolor=colors['background'], font_color=colors['label text'])
    fig_website.update_layout(title_x=0.5)
    
    return [unique_users, tot_questions, avg_mess_per_user, minimum_mess_per_user, maximum_mess_per_user, accuracy, precision, 
    percentage_picked_from_pick_list, fig_count_by_date, fig_cum_sum_by_date, fig_perf, fig_intent, fig_website, fig_hour, fig_browser]

@app.callback(
[Output('chatbots', 'value'),
Output('platform', 'value'),
Output('website_source', 'value')],
[Input('select_all_chatbots', 'n_clicks'),
Input('select_all_platform', 'n_clicks'),
Input('select_all_website', 'n_clicks')],
[State('chatbots', 'options'),
State('platform', 'options'),
State('website_source', 'options')])

def update_dropdown(btn1, btn2, btn3, feature_options_chatbots,feature_options_platform,feature_options_website):
    ctx = callback_context
    input_id = ctx.triggered[0]["prop_id"].split(".")[0]
    print(input_id)
    chatbot_select_all = [i['value'] for i in feature_options_chatbots]
    platform_select_all = [i['value'] for i in feature_options_platform]
    website_select_all = [i['value'] for i in feature_options_website]
    if input_id == 'select_all_chatbots':

        if btn1 % 2 != 0: ## Clear all options on even clicks
            chatbot_select_all = []
            #return []
        else: ## Select all options on odd clicks
            chatbot_select_all = [i['value'] for i in feature_options_chatbots]
            #return [i['value'] for i in feature_options_chatbots]
    elif input_id == 'select_all_platform':

        if btn2 % 2 != 0: ## Clear all options on even clicks
            platform_select_all = []
            #return []
        else: ## Select all options on odd clicks
            platform_select_all = [i['value'] for i in feature_options_platform]
            #return [i['value'] for i in feature_options_platform]
    elif input_id == 'select_all_website':

        if btn3 % 2 != 0: ## Clear all options on even clicks
            website_select_all = []
            #return []
        else: ## Select all options on odd clicks
            website_select_all = [i['value'] for i in feature_options_website]
    # else:
    #     raise PreventUpdate()
    return [chatbot_select_all, platform_select_all, website_select_all]



# Running the App

if __name__ == '__main__':
    app.run_server(host="0.0.0.0",port=8050,debug=False)