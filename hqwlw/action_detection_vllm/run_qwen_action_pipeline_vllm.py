# run_qwen_action_pipeline_vllm.py
from pathlib import Path
import os

from action_detection_vllm.pipeline_qwen_lzy import QwenPersonActionPipeline


def create_pipeline() -> QwenPersonActionPipeline:
    # vLLM 环境变量
    os.environ["VLLM_BASE_URL"] = "http://192.168.130.162:8010/v1"
    os.environ["VLLM_MODEL"]    = "/model/Qwen3-VL-2B-Instruct-FP8"

    os.environ["QWEN_MAX_NEW_TOKENS"] = "128"
    os.environ["QWEN_TEMPERATURE"]    = "0.0"
    os.environ["QWEN_TOP_P"]          = "0.9"

    pipeline = QwenPersonActionPipeline(
        grounding_model_id="/home/cs/leiheao/dataengine/grounding-dino/snapshots/a2bb814dd30d776dcf7e30523b00659f4f141c71",
        text_query="person. ",
        grounding_threshold=0.7,
        text_threshold=0.25,
        qwen_model_dir="/home/cs/leiheao/dataengine/Qwen3vl-2B",
        system_prompt_path="/workspace/action_detection_vllm/prompt/system.yaml",
        user_prompt_path="/workspace/action_detection_vllm/prompt/user.yaml",
        data_config="/home/cs/leiheao/dataengine/data_10_28_datav6.yaml",
        device="cuda:2",
        qwen_device="cuda",
        qwen_model_name="Qwen3-VL",
    )
    return pipeline