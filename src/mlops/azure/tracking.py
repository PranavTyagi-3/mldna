from azure.ai.ml import MLClient


def bootstrap_tracking_uri(subscription_id, resource_group, workspace_name, credential):
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    workspace = ml_client.workspaces.get(workspace_name)
    return workspace.mlflow_tracking_uri


def list_workspaces(subscription_id, credential):
    ml_client = MLClient(credential, subscription_id)
    return [(ws.name, ws.resource_group) for ws in ml_client.workspaces.list(scope="subscription")]
