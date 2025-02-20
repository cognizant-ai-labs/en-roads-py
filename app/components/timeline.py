"""
Timeline component.
"""
from collections import defaultdict

import numpy as np
from dash import html, Input, Output, State
import dash_bootstrap_components as dbc
import yaml

from app.components.component import Component
from enroadspy import load_input_specs


class TimelineComponent(Component):
    """
    Component handling generation of a timeline of actions taken.
    """
    def __init__(self, actions: list[str]):
        self.actions = [a for a in actions]

        # Pre-compute timeline actions for later when we show actions
        with open("app/timeline.yaml", "r", encoding="utf-8") as f:
            self.timeline = yaml.load(f, Loader=yaml.FullLoader)
        self.timeline_actions = []
        for action, details in self.timeline.items():
            self.timeline_actions.append(action)
            for d in details.values():
                self.timeline_actions.append(d)

        self.input_specs = load_input_specs()

    def _create_timeline_events(self,
                                action: str,
                                details: dict,
                                context_actions_dict: dict[str, float]) -> dict[int, str]:
        """
        Creates 0 or more timeline events for a given action.
        We have to manually handle electric standard being active and if this is the final carbon tax as they rely on
        other actions.
        We also manually handle if the action is a different type of bioenergy
        Returns a dict with the year as the key and the event as the value.
        """
        # Electric standard needs to be active
        if action == "_electric_standard_target" and not context_actions_dict["_electric_standard_active"]:
            return {}
        # If we're doing subsidy by feedstock ignore the default bio subsidy
        if action == "_source_subsidy_delivered_bio_boe" and context_actions_dict["_use_subsidies_by_feedstock"]:
            return {}
        # If we're not doing subsidy by feedstock ignore the specific feedstock subsidies
        if action in ["_wood_feedstock_subsidy_boe", "_crop_feedstock_subsidy_boe", "_other_feedstock_subsidy_boe"] \
           and not context_actions_dict["_use_subsidies_by_feedstock"]:
            return {}

        events = {}
        row = self.input_specs[self.input_specs["varId"] == action].iloc[0]
        name = row["varName"]
        decimal = np.ceil(-1 * np.log10(row["step"])).astype(int)
        value = context_actions_dict[action]

        start_year = context_actions_dict[details["start"]]

        # Carbon price phasing start date needs to be after previous carbon price phase end date
        if action == "_carbon_tax_final_target":
            initial_end = context_actions_dict["_carbon_tax_phase_1_start"] + \
                context_actions_dict["_carbon_tax_time_to_achieve_initial_target"]
            if start_year < initial_end:
                start_year = initial_end

        # Compute the stop year from the length if necessary
        if "stop" in details or "length" in details:
            stop_year = context_actions_dict[details["stop"]] if "stop" in details else start_year + \
                context_actions_dict[details["length"]]
            if start_year < stop_year:
                events[int(start_year)] = f"start {name}: {value:.{decimal}f}"
                events[int(stop_year)] = f"end {name}"
        else:
            events[int(start_year)] = f"{name}: {value:.{decimal}f}"

        return events

    def create_timeline(self, context_actions_dict: dict[str, float]) -> html.Div:
        """
        Creates a nice timeline of actions taken as well as the initial actions taken.
        """
        # Generate initial actions
        children = [html.H3("Initial Actions")]
        for action in context_actions_dict:
            if action not in self.timeline_actions and action in self.actions:
                input_spec = self.input_specs[self.input_specs["varId"] == action].iloc[0]
                val = context_actions_dict[action]
                if input_spec["kind"] == "slider":
                    formatting = input_spec["format"]
                    val_formatted = f"{val:{formatting}}"
                else:
                    val_formatted = "on" if val else "off"
                children.append(html.P(f"{input_spec['varName']}: {val_formatted}"))

        # Generate timeline
        timeline = defaultdict(list)
        for action, details in self.timeline.items():
            if action in context_actions_dict:
                events = self._create_timeline_events(action, details, context_actions_dict)
                for year, event in events.items():
                    timeline[year].append(event)

        # Sort the timeline by year so that we can display it in order
        children.append(html.H3("Timeline"))
        for year in sorted(timeline.keys()):
            children.append(html.H4(str(year)))
            for event in timeline[year]:
                children.append(html.P(event))

        return html.Div(children)

    def create_div(self) -> html.Div:
        """
        A button that produces a modal of the timeline.
        """
        div = html.Div(
            children=[
                dbc.Button("Show Actions", id="show-actions-button"),
                dbc.Modal(
                    id="actions-modal",
                    scrollable=True,
                    is_open=False,
                    children=[
                        dbc.ModalHeader(dbc.ModalTitle("Actions")),
                        dbc.ModalBody("These are the actions taken", id="actions-body")
                    ]
                )
            ]
        )
        return div

    def register_callbacks(self, app):
        """
        Registers callbacks that open and close the modal and update the text when a candidate is selected.
        """
        @app.callback(
            Output("actions-modal", "is_open"),
            Input("show-actions-button", "n_clicks"),
            State("actions-modal", "is_open")
        )
        def toggle_actions_modal(n_clicks: int, is_open: bool) -> bool:
            """
            Toggles the actions modal on and off.
            """
            if n_clicks:
                return True
            return is_open

        @app.callback(
            Output("actions-body", "children"),
            Output("show-actions-button", "disabled"),
            State("context-actions-store", "data"),
            Input("cand-link-select", "value")
        )
        def update_actions_body(context_actions_dicts: list[dict[str, float]], cand_idx: int) -> tuple[str, bool]:
            """
            Updates the body of the modal when a candidate is selected.
            """
            if cand_idx is not None:
                context_actions_dict = context_actions_dicts[cand_idx]
                # return self.timeline_component.create_timeline_div(context_actions_dict)
                return self.create_timeline(context_actions_dict), False
            return "Timeline goes here", False
