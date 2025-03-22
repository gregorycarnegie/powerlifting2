import logging

import dash_bootstrap_components as dbc
from dash import html

from services.config_service import config
from views.components import create_filter_sidebar, create_visualization_area, create_info_modal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_layout() -> html.Div:
    """
    Create the main application layout.
    
    Returns:
        Dash HTML Div containing the layout
    """
    app_name = config.get("app", "name") or "Powerlifting Data Visualization"
    
    return html.Div([
        # Header
        html.Div([
            html.H1(app_name, className="display-4 text-center my-4"),
            html.Div([
                dbc.Button("About", id="open-info-button", color="secondary", outline=True, size="sm", className="me-2"),
                html.A(
                    dbc.Button("GitHub", color="dark", outline=True, size="sm"),
                    href="https://github.com/yourusername/powerlifting-visualizer",
                    target="_blank"
                )
            ], className="d-flex justify-content-end mb-3")
        ], className="container"),
        
        # Main content
        dbc.Container([
            dbc.Row([
                # Left sidebar with filters
                create_filter_sidebar(),
                
                # Main visualization area
                create_visualization_area()
            ])
        ], fluid=True),
        
        # Footer
        html.Footer([
            html.Div([
                html.P([
                    "Data provided by ",
                    html.A("OpenPowerlifting", href="https://www.openpowerlifting.org/", target="_blank"),
                    ". Built with ",
                    html.A("Dash", href="https://dash.plotly.com/", target="_blank"),
                    " and ",
                    html.A("Polars", href="https://pola.rs/", target="_blank"),
                    "."
                ], className="text-center text-muted")
            ], className="container py-3")
        ], className="mt-5 pt-4 border-top"),
        
        # Modals
        create_info_modal()
    ])
