import dash
from dash import html

def create_dash_app(server):
    dash_app = dash.Dash(__name__, server=server, url_base_pathname="/")
    dash_app.layout = html.Div([
        html.H1("Dash Frontend"),
        html.P("This is served inside FastAPI")
    ])
    return dash_app
