def train_models(models, X_train, y_train):
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"[INFO] Trained: {name}")

    return trained