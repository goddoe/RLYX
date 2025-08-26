import re
from typing import List, Tuple


###############################################
# 1) 전처리(정규화) 유틸
###############################################
_PUNCT = re.compile(r"[.,!?;:·…""\"''']")

def normalize(text: str, *, remove_space: bool = False) -> str:
    """
    - 소문자화
    - 구두점 제거
    - 연속 공백 → ' '  (remove_space=True 면 전부 제거)
    """
    text = text.lower()
    text = _PUNCT.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_space:
        text = text.replace(" ", "")
    return text

###############################################
# 2) Levenshtein distance
###############################################
def _levenshtein(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:  return m
    if m == 0:  return n
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j]+1,      # 삭제
                           dp[i][j-1]+1,      # 삽입
                           dp[i-1][j-1]+cost) # 대치
    return dp[n][m]

###############################################
# 3) 지표
###############################################
def _wer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref_words = ref.split()
    hyp_words = hyp.split()
    err = _levenshtein(ref_words, hyp_words)
    return (err/len(ref_words) if ref_words else 0.0, err, len(ref_words))

def _cer(ref: str, hyp: str) -> Tuple[float, int, int]:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    err = _levenshtein(ref_chars, hyp_chars)
    return (err/len(ref_chars) if ref_chars else 0.0, err, len(ref_chars))