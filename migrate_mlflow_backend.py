import mlflow
from mlflow.entities import Param
from tqdm.auto import tqdm


def check_if_run_exists(from_run, from_client, to_runs, to_experiment_id, to_client):
    possible_runs = [run for run in to_runs if run.info.run_name == from_run.info.run_name]
    run_exists = False
    for run in possible_runs:
        if run.data.metrics == from_run.data.metrics and run.data.params == from_run.data.params:
            run_exists = True
    if not run_exists:
        create_run_from(from_run, from_client, to_experiment_id, to_client)


def create_experiment_from(from_experiment, to_client):
    name = from_experiment.name
    tags = from_experiment.tags
    return to_client.create_experiment(name=name, tags=tags)


def create_run_from(from_run, from_client, to_experiment_id, to_client):
    start_time = from_run.info.start_time
    tags = from_run.data.tags
    run_name = from_run.info.run_name
    new_run = to_client.create_run(experiment_id=to_experiment_id, start_time=start_time,
                                   tags=tags, run_name=run_name)
    params = [Param(key, value) for key, value in from_run.data.params.items()]
    metrics_keys = from_run.data.metrics.keys()
    metrics = []
    for metric_key in metrics_keys:
        metrics = metrics + from_client.get_metric_history(from_run.info.run_id, metric_key)
    to_client.log_batch(new_run.info.run_id, metrics=metrics, params=params)
    to_client.set_terminated(new_run.info.run_id, end_time=from_run.info.end_time, status=from_run.info.status)


def migrate_mlflow_backend(from_tracking_uri, to_tracking_uri):
    from_client = mlflow.client.MlflowClient(tracking_uri=from_tracking_uri)
    to_client = mlflow.client.MlflowClient(tracking_uri=to_tracking_uri)
    from_experiments = from_client.search_experiments()
    to_experiments = to_client.search_experiments()
    pbar_experiment = tqdm(from_experiments)
    for from_experiment in pbar_experiment:
        msg = f'Migrating experiment: {from_experiment.name}'
        pbar_experiment.set_description_str(msg)
        if from_experiment.name not in [exp.name for exp in to_experiments]:
            to_experiment_id = create_experiment_from(from_experiment, to_client)
        else:
            to_experiment_id = to_client.get_experiment_by_name(from_experiment.name).experiment_id
        from_runs = from_client.search_runs(from_experiment.experiment_id, max_results=50000)
        to_runs = to_client.search_runs(to_experiment_id, max_results=50000)
        pbar = tqdm(from_runs)
        for from_run in pbar:
            msg = f'Migrating run: {from_run.info.run_name} id {from_run.info.run_id}'
            pbar.set_description_str(msg)
            if from_run.info.run_name in [run.info.run_name for run in to_runs]:
                check_if_run_exists(from_run, from_client, to_runs, to_experiment_id, to_client)
            else:
                create_run_from(from_run, from_client, to_experiment_id, to_client)
