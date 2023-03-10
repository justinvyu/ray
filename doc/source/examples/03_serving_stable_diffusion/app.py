from io import BytesIO

import ray
from ray import serve
from fastapi import FastAPI
from fastapi.responses import Response


NUM_NODES = len(ray.nodes())

app = FastAPI()


@serve.deployment(num_replicas=1, route_prefix="/")
@serve.ingress(app)
class APIIngress:
    def __init__(self, diffusion_model_handle) -> None:
        self.handle = diffusion_model_handle

    @app.get(
        "/imagine",
        responses={200: {"content": {"image/png": {}}}},
        response_class=Response,
    )
    async def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = await (await self.handle.generate.remote(prompt, img_size=img_size))

        file_stream = BytesIO()
        image.save(file_stream, "PNG")
        return Response(content=file_stream.getvalue(), media_type="image/png")


def get_runtime_env():
    # Read in the requirements.txt and install these packages in the environment
    # of each worker.
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
    return {"pip": requirements}


@serve.deployment(
    ray_actor_options={"num_gpus": 1, "runtime_env": get_runtime_env()},
    autoscaling_config={"min_replicas": NUM_NODES, "max_replicas": NUM_NODES},
)
class StableDiffusionV2:
    def __init__(self):
        # Delay imports, since packages will only be installed on the replicas
        import torch
        from diffusers import EulerDiscreteScheduler, StableDiffusionPipeline

        model_id = "stabilityai/stable-diffusion-2"

        scheduler = EulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16
        )
        self.pipe = self.pipe.to("cuda")

    def generate(self, prompt: str, img_size: int = 512):
        assert len(prompt), "prompt parameter cannot be empty"

        image = self.pipe(prompt, height=img_size, width=img_size).images[0]

        return image


entrypoint = APIIngress.bind(StableDiffusionV2.bind())

# Run the script with `serve run app:entrypoint` to start the serve application
