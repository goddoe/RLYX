import os

import torch
from ray import serve
from vllm import LLM, SamplingParams
from vllm.worker.worker import Worker
from starlette.requests import Request

MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-0.5B")
# MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "Qwen/Qwen2.5-3B")
# MODEL_NAME_OR_PATH = os.environ.get("MODEL_NAME_OR_PATH", "HuggingFaceTB/SmolLM2-360M")
NUM_INFER_WORKERS = os.environ.get("NUM_INFER_WORKERS", 8)
print(f"MODEL_NAME_OR_PATH: {MODEL_NAME_OR_PATH}")
print(f"NUM_INFER_WORKERS: {NUM_INFER_WORKERS}")


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes) 
    and vLLM workers.
    """
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WrappedWorker(Worker):

    def init_weight_update_group(self, master_address, master_port,
                                 rank, world_size):
        from vllm.distributed.parallel_state import get_world_group

        print(f"{get_world_group().rank=}, {rank=}")
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )

    def update_weight(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])
        del weight


@serve.deployment(num_replicas=NUM_INFER_WORKERS,
                  ray_actor_options={"num_gpus": 1})
class InferenceWorker:
    def __init__(self):
        self.llm = LLM(model=MODEL_NAME_OR_PATH,
                       enforce_eager=True,
                       worker_cls=WrappedWorker,
                       dtype="half"
                       )
        self.worker_id = os.getpid()

    def init_weight_update_group(self, master_address, master_port, rank, world_size):
        self.llm.collective_rpc("init_weight_update_group",
                                args=(master_address, master_port, rank, world_size))
        return "Weight update group initialized."

    def update_weight(self, name, dtype, shape):
        self.llm.collective_rpc("update_weight",
                                args=(name, dtype, shape))
        return "Weight updated."

    def who_you_are(self, val: int) -> str:
        return f"Worker {self.worker_id} processed value: {val}"

    def generate_text(self, prompts, sample_params=None):
        if sample_params is None:
            sample_params = {'temperature': 0.7,
                             'max_tokens': 512,
                             'n': 4}

        outputs = self.llm.generate(prompts,
                                    SamplingParams(**sample_params))

        return {"text": [[out.text for out in output.outputs]
                            for output in outputs],
                "token_ids": [[list(out.token_ids) for out in output.outputs]
                            for output in outputs]
                }


    async def __call__(self, http_request: Request):
        data = await http_request.json()
        prompts = data.get("prompts", [])
        results = self.generate_text(prompts)
        return results

# Ray Serve App
inference_app = InferenceWorker.bind()
