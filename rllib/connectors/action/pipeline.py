from typing import Any, List

from ray.rllib.connectors.connector import (
    ActionConnector,
    Connector,
    ConnectorContext,
    ConnectorPipeline,
    get_connector,
    register_connector,
)
from ray.rllib.utils.typing import ActionConnectorDataType
from ray.util.annotations import PublicAPI


@PublicAPI(stability="alpha")
class ActionConnectorPipeline(ConnectorPipeline, ActionConnector):
    def __init__(self, ctx: ConnectorContext, connectors: List[Connector]):
        super().__init__(ctx, connectors)

    def __call__(self, ac_data: ActionConnectorDataType) -> ActionConnectorDataType:
        for c in self.connectors:
            ac_data = c(ac_data)
        return ac_data

    def to_state(self):
        return ActionConnectorPipeline.__name__, [c.to_state() for c in self.connectors]

    @staticmethod
    def from_state(ctx: ConnectorContext, params: List[Any]):
        assert (
            type(params) == list
        ), "ActionConnectorPipeline takes a list of connector params."
        connectors = [get_connector(ctx, name, subparams) for name, subparams in params]
        return ActionConnectorPipeline(ctx, connectors)


register_connector(ActionConnectorPipeline.__name__, ActionConnectorPipeline)
