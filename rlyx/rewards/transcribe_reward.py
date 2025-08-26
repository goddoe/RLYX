import numpy as np
from .utils import normalize, _wer, _cer
from rlyx.registries import REWARD_REGISTRY


@REWARD_REGISTRY.register("transcribe_reward")
def transcribe_reward_func(pred_text: str,
                           gold_text: str,
                           rich: bool = False,
                           beta: float = 2.0,
                           **kwargs):
    # ---------- 정규화 ----------
    gold_w = normalize(gold_text)          # 구두점 제거, 공백 유지
    pred_w = normalize(pred_text)
    gold_c = normalize(gold_text, remove_space=True)  # 공백도 제거
    pred_c = normalize(pred_text, remove_space=True)

    # ---------- 오류율 ----------
    wer_val, wer_err, wer_len = _wer(gold_w, pred_w)
    cer_val, cer_err, cer_len = _cer(gold_c, pred_c)

    # ---------- 보상 ----------
    error = (wer_val + cer_val) / 2.0
    tts_reward = np.exp(-beta * error)       # 0–1 사이
    tts_reward = min(tts_reward, 1.0)        # 안전 클램프
    if tts_reward < 0.2:                     # 너무 낮으면 0
        tts_reward = 0.0

    # ---------- 로그 ----------
    print(
        "--------------------------------------------------------\n"
        f"gold_text: {gold_text}\n"
        f"pred_text: {pred_text}\n"
        f"WER: {wer_val:.3f} (errors: {wer_err}, length: {wer_len})\n"
        f"CER: {cer_val:.3f} (errors: {cer_err}, length: {cer_len})\n"
        f"tts_reward: {tts_reward:.6f}"
    )

    if rich:
        return dict(
            tts_reward=tts_reward,
            wer_value=wer_val, cer_value=cer_val,
            wer_err=wer_err, cer_err=cer_err,
            wer_len=wer_len, cer_len=cer_len
        )
    return tts_reward


