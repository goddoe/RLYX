__all__ = ["cvt_audio_and_transcribe_completion"]

from typing import List, Tuple
import re
import base64
from ray import serve
from .decoder_utils import decode
from concurrent.futures import ThreadPoolExecutor


whisper_handle = None


def batch_cvt_audio_and_transcribe_completion(completion_list, max_workers=8):
    results   = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in executor.map(cvt_audio_and_transcribe_completion,
                                completion_list):
            results.append(res)
    return results


def cvt_audio_and_transcribe_completion(completion):

    global whisper_handle
    if whisper_handle is None:
        whisper_handle = serve.get_deployment_handle("WhisperWorker",
                                                     app_name="WhisperService")

    def unit_token_to_idx_list(model_unit: str) -> List[int]:
        """
        Return a unit sequence from unit tokens
        Input: <|unitXXXA|><|unitXXXB|><|unitXXXC|><|unitXXXD|>
        Output: [XXXA, XXXB, XXXC, XXXD]
        """
        out = [int(i) for i in re.findall(r"\d+", model_unit) if str(i).isdigit() and (0<=int(i)<=9999)]
        return out

    unit_ids = unit_token_to_idx_list(completion)
    if not unit_ids:
        return "", None

    speech_bin = decode(unit_ids, "unit-bigvgan-vat1", ref_speaker="fkms")
    audio_base64 = base64.b64encode(speech_bin).decode('utf-8')
    
    result = whisper_handle.transcribe_from_base64.remote(
                                          audio_base64=audio_base64,
                                          format="wav",
                                          language="ko").result()
    """
    {
      'text': ' 운동에 대해 이야기해 볼까?',
      'segments': [
        {
          'id': 0,
          'seek': 0,
          'start': 0.0,
          'end': 3.0,
          'text': ' 운동에 대해 이야기해 볼까?',
          'tokens': [
            50364,
            33541,
            1517,
            48374,
            37576,
            5302,
            18001,
            3294,
            30,
            50514
          ],
          'temperature': 0.0,
          'avg_logprob': -0.9614426872947,
          'compression_ratio': 0.8043478260869565,
          'no_speech_prob': 0.17980362474918365
        }
      ],
      'language': 'ko',
      'worker_id': 1426
    }
    """
    pred_text = result['text']
    return pred_text, speech_bin


