import argparse

import keras
import tensorflow as tf

import ray
from ray import train
from ray.air.integrations.keras import ReportCheckpointCallback
from ray.data.preprocessors import Concatenator
from ray.train import Result, ScalingConfig
from ray.train.tensorflow import TensorflowTrainer


def build_model() -> keras.Model:
    model = keras.Sequential(
        [
            tf.keras.layers.Input(shape=(100,)),
            tf.keras.layers.Dense(10),
            tf.keras.layers.Dense(1),
        ]
    )
    return model


def train_func(config: dict):
    print("in train func...")
    batch_size = config.get("batch_size", 64)
    epochs = config.get("epochs", 3)

    strategy = tf.distribute.MultiWorkerMirroredStrategy()
    with strategy.scope():
        print("!! building model...")
        # Model building/compiling need to be within `strategy.scope()`.
        multi_worker_model = build_model()
        print("!! compiling...")
        multi_worker_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=config.get("lr", 1e-3)),
            loss=tf.keras.losses.MeanAbsoluteError(),
            metrics=[tf.keras.metrics.MeanSquaredError()],
        )

    print("!! built model")

    dataset = train.get_dataset_shard("train")

    results = []
    for _ in range(epochs):
        # tf_dataset = dataset.to_tf(
        #     feature_columns="x", label_columns="y", batch_size=batch_size
        # )
        print("in train train loop...")
        for batch in dataset.iter_batches(batch_size=batch_size):
            print("in train train loop batch...")
            # , callbacks=[ReportCheckpointCallback()]
            multi_worker_model.train_on_batch(batch["x"], batch["y"])

        # results.append(history.history)
    print("training done...")
    return results


def train_tensorflow_regression(num_workers: int = 2, use_gpu: bool = False) -> Result:
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/regression.csv")
    preprocessor = Concatenator(exclude=["", "y"], output_column_name="x")
    dataset = preprocessor.fit_transform(dataset)

    config = {"lr": 1e-3, "batch_size": 32, "epochs": 4}
    scaling_config = ScalingConfig(num_workers=num_workers, use_gpu=use_gpu)
    trainer = TensorflowTrainer(
        train_loop_per_worker=train_func,
        train_loop_config=config,
        scaling_config=scaling_config,
        datasets={"train": dataset},
    )
    results = trainer.fit()
    print(results.metrics)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address", required=False, type=str, help="the address to use for Ray"
    )
    parser.add_argument(
        "--num-workers",
        "-n",
        type=int,
        default=2,
        help="Sets number of workers for training.",
    )
    parser.add_argument(
        "--use-gpu", action="store_true", default=False, help="Enables GPU training"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Finish quickly for testing.",
    )

    args, _ = parser.parse_known_args()

    if args.smoke_test:
        # 2 workers, 1 for trainer, 1 for datasets
        num_gpus = args.num_workers if args.use_gpu else 0
        ray.init(num_cpus=4, num_gpus=num_gpus)
        result = train_tensorflow_regression(num_workers=2, use_gpu=args.use_gpu)
    else:
        ray.init(address=args.address)
        result = train_tensorflow_regression(
            num_workers=args.num_workers, use_gpu=args.use_gpu
        )
    print(result)
