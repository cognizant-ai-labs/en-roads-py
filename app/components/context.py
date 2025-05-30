"""
Component displaying the context and handling its related functions.
"""
from dash import dcc, html, Input, Output, ALL
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd

from app.components.component import Component
from enroadspy import load_input_specs, id_to_name, id_to_varid

# Constants used for SSPs
SSP_VALUES = [
    "SSP1",
    "SSP2",
    "SSP3",
    "SSP4",
    "SSP5"
]
SSP_LABELS = [
    "SSP1: Sustainable Development",
    "SSP2: Middle of the Road",
    "SSP3: Regional Rivalry",
    "SSP4: Inequality",
    "SSP5: Fossil-Fueled"
]
SSP_COLORS = [
    "#1a8955",
    "#688114",
    "#dc3848",
    "#aa6900",
    "#aa6900"
]


class ContextComponent(Component):
    """
    Component in charge of displaying the preset contexts in a scatter and the corresponding context sliders.
    Adjusts context sliders to match selected preset and displays a blurb about the selected SSP.
    """
    def __init__(self, context: list[int]):

        self.context = list(context)

        # Round context df here instead of automatically by Dash so that we know for sure how it's rounding.
        # NOTE: context_df uses varIds from old experiments instead of id so we need to convert whenever we access it.
        self.context_df = pd.read_csv("experiments/scenarios/gdp_context.csv")
        input_specs = load_input_specs()
        for col in self.context:
            row = input_specs[input_specs["id"] == col].iloc[0]
            step = row["step"]
            decimals = -1 * int(np.log10(step))
            varid = id_to_varid(col, input_specs)
            self.context_df[varid] = self.context_df[varid].round(decimals)

        self.ssp_df = pd.read_csv("app/ssps.csv")

    def create_context_sliders(self) -> html.Div:
        """
        Creates labels aligned with sliders for the context variables.
        """
        input_specs = load_input_specs()
        sliders = []
        for i, context_id in enumerate(self.context):
            row = input_specs[input_specs["id"] == context_id].iloc[0]
            label_slider = dbc.Row([
                # TODO: It's annoying that bootstrap only does 12 columns and 5/7 split doesn't look nice.
                dbc.Col(
                    width=6,
                    children=[html.Label(id_to_name(context_id, input_specs))]
                ),
                dbc.Col(
                    width=6,
                    children=[
                        dcc.Slider(
                            id={"type": "context-slider", "index": i},
                            min=row["minValue"],
                            max=row["maxValue"],
                            step=row["step"],
                            value=row["defaultValue"],
                            tooltip={"placement": "bottom"},
                            marks=None,
                        )
                    ]
                )
            ])
            sliders.append(label_slider)

        # TODO: Fix the widths to be responsive
        sliders_div = html.Div(
            children=sliders
        )
        return sliders_div

    def create_ssp_dropdown(self) -> dcc.Dropdown:
        """
        Creates dropdown to select an SSP
        """
        options = []
        for value, label, color in zip(SSP_VALUES, SSP_LABELS, SSP_COLORS):
            options.append({"label": html.Span([label], style={"color": color}), "value": value})

        dropdown = dcc.Dropdown(
            id="ssp-dropdown",
            options=options,
            placeholder="Select a Scenario"
        )
        return dropdown

    def create_div(self) -> html.Div:
        """
        Creates big context div for larger demos.
        We create a card with a dropdown to select the SSP and sliders to adjust the selected SSP.
        The description is shown alongside the sliders.
        """
        sliders_div = self.create_context_sliders()
        dropdown_div = self.create_ssp_dropdown()
        div = dbc.Card(
            color="secondary",
            outline=True,
            children=[
                dbc.CardHeader(
                    children=[
                        html.H3("1. Select Context Scenario", className="text-center"),
                        dropdown_div
                    ]
                ),
                dbc.CardBody(
                    children=[
                        dbc.Row(
                            children=[
                                dbc.Col([sliders_div]),
                                dbc.Col(
                                    children=[
                                        html.Div(
                                            id="ssp-desc",
                                            children=[html.H4("Select a Scenario")],
                                        )
                                    ]
                                )
                            ]
                        ),
                        dbc.Row(
                            justify="center",
                            children=dbc.Button(
                                "Generate AI Policies for Scenario",
                                id="presc-button",
                                className="me-1 mb-2 w-50",
                                n_clicks=0
                            )
                        )
                    ]
                )
            ]
        )
        return html.Div(div)

    def construct_ssp_desc(self, ssp_idx: int) -> html.Div:
        """
        Constructs the description text for the SSP.
        We use a shortened description then add a link to the SSPs website.
        """
        ssp_row = self.ssp_df.iloc[ssp_idx]
        text = ssp_row["text"].split(".")[0] + "... "
        div = html.Div([
            html.H4([
                ssp_row["ssp"] + ": " + ssp_row["description"] + " ",
                html.I(className=f"bi bi-{ssp_row['icon']}")
            ]),
            html.P(text)
        ])
        return div

    def register_callbacks(self, app):
        """
        Registers app's callbacks.
        """
        @app.callback(
            Output("ssp-desc", "children"),
            Input({"type": "context-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def update_ssp_desc(context_values):
            """
            If we click on a point, show the description of the SSP.
            If there is no click data, that means we updated the sliders, so remove the description.
            """
            input_specs = load_input_specs()
            varids = [id_to_varid(col, input_specs) for col in self.context]
            match = self.context_df[varids].eq(context_values).all(axis=1)

            # If a match is found, retrieve the row, otherwise handle the case where it isn't found
            if match.any():
                ssp_idx = match.idxmax()
                return self.construct_ssp_desc(ssp_idx)
            else:
                return html.Div(
                    children=[
                        html.H4("Custom Scenario Selected")
                    ]
                )

        @app.callback(
            Output("presc-button", "disabled", allow_duplicate=True),
            Output("presc-button", "children", allow_duplicate=True),
            Output("presc-button", "color", allow_duplicate=True),
            Input("presc-button", "n_clicks"),
            prevent_initial_call=True
        )
        def disable_button(n_clicks):
            """
            Disables the button after it is clicked and displays a loading message.
            """
            return n_clicks > 0, "Please wait...", "warning"

        @app.callback(
            Output("presc-button", "disabled", allow_duplicate=True),
            Output("presc-button", "children", allow_duplicate=True),
            Output("presc-button", "color", allow_duplicate=True),
            Input("reset-button", "disabled"),
            prevent_initial_call=True
        )
        def enable_button(_):
            """
            Enables the button when the filtering is done and resets it.
            """
            return False, "Generate AI Policies for Scenario", "primary"

        @app.callback(
            Output({"type": "context-slider", "index": ALL}, "value", allow_duplicate=True),
            Input("ssp-dropdown", "value"),
            prevent_initial_call=True
        )
        def select_ssp_dropdown(value):
            """
            When we select a dropdown value, update the sliders.
            If we clear the dropdown, reset the sliders to the default values.
            """
            input_specs = load_input_specs()

            # If the value is a scenario, find the corresponding row in context_df.
            scenario = f"{value}-Baseline"
            if scenario in self.context_df["scenario"].unique():
                row = self.context_df[self.context_df["scenario"] == scenario].iloc[0]
                return [row[id_to_varid(self.context[i], input_specs)] for i in range(4)]

            # Default case: if the scenario is not found just get the default values
            return [input_specs[input_specs["id"] == self.context[i]]["defaultValue"].iloc[0] for i in range(4)]

        @app.callback(
            Output("ssp-dropdown", "value", allow_duplicate=True),
            Input({"type": "context-slider", "index": ALL}, "value"),
            prevent_initial_call=True
        )
        def clear_dropdown(context_values):
            """
            If the context sliders don't match any of the SSPs, reset the dropdown.
            """
            input_specs = load_input_specs()
            varids = [id_to_varid(col, input_specs) for col in self.context]
            match = self.context_df[varids].eq(context_values).all(axis=1)

            # If a match is found, retrieve the row, otherwise handle the case where it isn't found
            if match.any():
                ssp_idx = match.idxmax()
                return SSP_VALUES[ssp_idx]
