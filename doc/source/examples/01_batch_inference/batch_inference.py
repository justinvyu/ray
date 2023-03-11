import torch
import numpy as np
from torchvision import transforms
from typing import Dict

import ray
from ray.data.datasource.partitioning import Partitioning


ray.init()
NUM_NODES = len(ray.nodes())

# 1. Replace this to load your own data with Ray Data!
# Start
s3_uri = "s3://anonymous@air-example-data-2/imagenette2/val/"
partitioning = Partitioning("dir", field_names=["class"], base_dir=s3_uri)
ds = ray.data.read_images(
    s3_uri, size=(256, 256), partitioning=partitioning, mode="RGB"
)
# End


def preprocess(batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    # 2. Replace this with your own custom preprocessing logic!
    # Start
    def to_tensor(batch: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(batch, dtype=torch.float)
        # (B, H, W, C) -> (B, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        # [0., 255.] -> [0., 1.]
        tensor = tensor.div(255)
        return tensor

    transform = transforms.Compose(
        [
            transforms.Lambda(to_tensor),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return {"image": transform(batch["image"]).numpy()}
    # End


ds = ds.map_batches(fn=preprocess, batch_format="numpy")


class PredictCallable:
    def __init__(self):
        # 3. Load your model in a custom `Callable` class that will perform inference!
        # Replace this with your own model initialization:
        # Start
        from torchvision import models

        self.model = models.resnet152(pretrained=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # End

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Replace this with your own model inference logic:
        # Start
        input_data = torch.as_tensor(batch["image"], device=self.device)
        with torch.no_grad():
            result = self.model(input_data)
        return {"predictions": result.cpu().numpy()}
        # End


# 4. Finally, perform batch prediction!
predictions = ds.map_batches(
    PredictCallable,
    batch_size=128,
    compute=ray.data.ActorPoolStrategy(min_size=NUM_NODES, max_size=NUM_NODES),
    num_gpus=1,
    batch_format="numpy",
)

# 5. Save predictions to the local filesystem (sharded between multiple files)
num_shards = 3
predictions.repartition(num_shards).write_parquet("local:///tmp/predictions")

print(predictions.take(2))
print("Predictions saved to `/tmp/predictions`!")
