from pymilo.streaming import PymiloClient, Compression, CommunicationProtocol
from sklearn.linear_model import LinearRegression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario4(compression_method, communication_protocol):
    """
    Test access control management features between multiple clients.

    This scenario tests:
    1. Client registration/deregistration
    2. Model registration/deregistration
    3. get_ml_models
    4. grant_access / revoke_access
    5. get_allowance / get_allowed_models
    """
    x_train, y_train, _, _ = prepare_simple_regression_datasets()

    # Create and train a model locally
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)

    # Initialize client_a (model owner)
    client_a = PymiloClient(
        model=linear_regression,
        mode=PymiloClient.Mode.LOCAL,
        compressor=Compression[compression_method],
        server_url="127.0.0.1:8500",
        communication_protocol=CommunicationProtocol[communication_protocol],
    )

    # Initialize client_b (will be granted access)
    client_b = PymiloClient(
        mode=PymiloClient.Mode.LOCAL,
        compressor=Compression[compression_method],
        server_url="127.0.0.1:8500",
        communication_protocol=CommunicationProtocol[communication_protocol],
    )

    # 1. Register both clients
    client_a.register()
    client_b.register()

    assert client_a.client_id is not None, "client_a registration failed"
    assert client_b.client_id is not None, "client_b registration failed"
    assert client_a.client_id != client_b.client_id, "clients should have different IDs"

    # 2. Register model for client_a
    client_a.register_ml_model()
    assert client_a.ml_model_id is not None, "model registration failed"

    # 3. Test get_ml_models
    models_a = client_a.get_ml_models()
    assert client_a.ml_model_id in models_a, "registered model should appear in get_ml_models"

    # 4. Upload model from client_a
    client_a.upload()

    # 5. Grant access from client_a to client_b (uses client_a.ml_model_id implicitly)
    grant_result = client_a.grant_access(client_b.client_id)
    assert grant_result is True, "grant_access should succeed"

    # 6. Verify allowance updated
    allowance = client_a.get_allowance()
    assert isinstance(allowance, dict), "get_allowance should return a dict"
    assert client_b.client_id in allowance, "client_b should be in allowance after grant"
    assert client_a.ml_model_id in allowance[client_b.client_id], "model should be in allowance"

    # 7. Test get_allowed_models (from client_b's perspective)
    allowed_models = client_b.get_allowed_models(client_a.client_id)
    assert client_a.ml_model_id in allowed_models, "model should be in allowed_models"

    # 8. Revoke access (uses client_a.ml_model_id implicitly)
    revoke_result = client_a.revoke_access(client_b.client_id)
    assert revoke_result is True, "revoke_access should succeed"

    # 9. Verify allowance updated after revoke
    allowed_models_after_revoke = client_b.get_allowed_models(client_a.client_id)
    assert client_a.ml_model_id not in allowed_models_after_revoke, "model should not be in allowed_models after revoke"

    # 10. Test model deregistration
    models_before_deregister = client_a.get_ml_models()
    client_a.deregister_ml_model()
    models_after_deregister = client_a.get_ml_models()
    assert len(models_after_deregister) == len(models_before_deregister) - 1, "model count should decrease after deregister"

    # 11. Test client deregistration
    client_b.deregister()
    client_a.deregister()

    # 12. Clean up WebSocket connections if applicable
    if hasattr(client_a._communicator, 'close'):
        client_a._communicator.close()
    if hasattr(client_b._communicator, 'close'):
        client_b._communicator.close()

    return 0
