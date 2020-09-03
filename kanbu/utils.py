from sklearn.model_selection import GridSearchCV, cross_validate


def evaluate_learners(models, X, y):
    cross_validations = [
        cross_validate(model, X, y, return_train_score=True) for model in models
    ]

    return (
        list(map(lambda x: x["train_score"], cross_validations)),
        list(map(lambda x: x["test_score"], cross_validations)),
    )


def create_grid_searches(models, params, X, y):
    return [
        GridSearchCV(model, param, return_train_score=True).fit(X, y)
        for model, param in zip(models, params)
    ]
