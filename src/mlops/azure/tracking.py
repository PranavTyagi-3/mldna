from azure.ai.ml import MLClient


def bootstrap_tracking_uri(subscription_id, resource_group, workspace_name, credential):
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
    workspace = ml_client.workspaces.get(workspace_name)
    return workspace.mlflow_tracking_uri
