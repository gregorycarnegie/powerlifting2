import logging

import dash_bootstrap_components as dbc
from dash import dcc, html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def lift_tabs(*lifts: str, width: int = 12) -> list[dbc.Tab]:
    """
    Create tabs for different lifts.

    Args:
        *lifts: Lift names
        width: Column width

    Returns:
        List of Dash Bootstrap tabs
    """
    return [dbc.Tab(label=lift.capitalize(), children=[
        dbc.Row([dbc.Col([dcc.Graph(id=f'{lift}-histogram')], width=width)]),
        dbc.Row([dbc.Col([dcc.Graph(id=f'{lift}-wilks-histogram')], width=width)]),
        dbc.Row([dbc.Col([dcc.Graph(id=f'{lift}-scatter')], width=width)]),
        dbc.Row([dbc.Col([dcc.Graph(id=f'{lift}-wilks-scatter')], width=width)])
        ]) for lift in lifts]

def input_groups(
        *lifts: str,
        input_type: str = 'text',
        minimum: int = 0,
        step: float = 0.1,
        placeholder: str = 'kg',
        input_mode: str = 'numeric'
) -> list[dbc.InputGroup]:
    """
    Create input groups for different lifts.

    Args:
        *lifts: Lift names
        input_type: Input type
        minimum: Minimum value
        step: Step value
        placeholder: Placeholder text
        input_mode: Input mode
        pattern: Input pattern

    Returns:
        List of Dash Bootstrap input groups
    """
    return [dbc.InputGroup([
        dbc.InputGroupText(f"{lift.capitalize()}"),
        dbc.Input(
            id=f'{lift}-input',
            type=input_type,
            min=minimum,
            step=step,
            placeholder=placeholder,
            inputMode=input_mode
        )
    ], className="mb-2") for lift in lifts]

def checklist_options(*options: str) -> list[dict]:
    """
    Create checklist options.

    Args:
        *options: Option labels/values

    Returns:
        List of options dictionaries
    """
    return [{'label': opt, 'value': opt} for opt in options]

def create_filter_sidebar() -> dbc.Col:
    """
    Create the filter sidebar component.

    Returns:
        Dash Bootstrap Column with filter controls
    """
    return dbc.Col([
        html.Div([
            html.H4("Filter Options", className="mb-3"),
            html.Label("Sex:"),
            dbc.RadioItems(
                id='sex-filter',
                options=[
                    {'label': 'Male', 'value': 'M'},
                    {'label': 'Female', 'value': 'F'},
                    {'label': 'All', 'value': 'All'}
                ],
                value='M',
                className="mb-3",
                inline=True
            ),
            html.Label("Units:"),
            dbc.RadioItems(
                id='units',
                options=[
                    {'label': 'Metric (kg)', 'value': 'metric'},
                    {'label': 'Imperial (lbs)', 'value': 'imperial'}
                ],
                value='metric',
                className="mb-3",
                inline=True
            ),
            html.Label("Weight Class:"),
            dcc.Dropdown(
                id='weight-class-dropdown',
                options=[],  # Will be populated by callback
                value='all',
                clearable=False,
                className="mb-3"
            ),
            html.Label("Equipment:"),
            dbc.Checklist(
                id='equipment-filter',
                options=checklist_options('Raw', 'Wraps', 'Single-ply', 'Multi-ply', 'Unlimited', 'Straps'),
                value=['Raw'],
                className="mb-3"
            ),
            html.Label("Your Lifts:"),
            html.Div(input_groups("squat", "bench", "deadlift", "bodyweight")),
            # html.Div(id='bodyweight-input-trigger', style={'display': 'none'}),
            dbc.Button(
                "Update Visualizations",
                id="update-button",
                color="primary",
                className="w-100 mb-3"
            ),
            html.Div([
                html.P(id="last-updated-text", className="text-muted small")
            ])
        ], className="p-3 border rounded")
    ], width=3)

def create_visualization_area() -> dbc.Col:
    """
    Create the visualization area component.

    Returns:
        Dash Bootstrap Column with visualization tabs
    """
    return dbc.Col([
        dbc.Tabs(lift_tabs("squat", "bench", "deadlift", "total")),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.A(
                        dbc.Button("Share to Twitter", color="info", className="mr-2"),
                        id="share-twitter",
                        href="#",
                        target="_blank"
                    ),
                    html.A(
                        dbc.Button("Share to Facebook", color="primary", className="mr-2"),
                        id="share-facebook",
                        href="#",
                        target="_blank"
                    ),
                    html.A(
                        dbc.Button("Share to Instagram", color="danger"),
                        id="share-instagram",
                        href="#",
                        target="_blank"
                    )
                ], className="mt-3 d-flex justify-content-center gap-2")
            ], width=12)
        ])
    ], width=9)

def create_info_modal() -> dbc.Modal:
    """
    Create an information modal dialog.

    Returns:
        Dash Bootstrap Modal
    """
    return dbc.Modal(
        [
            dbc.ModalHeader("About This Application"),
            dbc.ModalBody([
                html.P([
                    "This application visualizes powerlifting data from the ",
                    html.A("OpenPowerlifting", href="https://www.openpowerlifting.org/", target="_blank"),
                    " database. You can filter by sex, equipment type, and weight class to see how your lifts compare."
                ]),
                html.P("Enter your lift numbers and bodyweight to see where you stand relative to other lifters."),
                html.P("The visualizations include:"),
                html.Ul([
                    html.Li("Histograms showing the distribution of lift values"),
                    html.Li("Scatter plots showing the relationship between bodyweight and lift values"),
                    html.Li("Both raw values and Wilks scores (which normalize for bodyweight differences)")
                ]),
                html.P("Data is updated regularly from the OpenPowerlifting database.")
            ]),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-info-modal", className="ms-auto")
            ),
        ],
        id="info-modal",
        size="lg",
    )
