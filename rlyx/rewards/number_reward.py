import re
from rlyx.registries import REWARD_REGISTRY


# ────────────────────────────────
# 0) 전처리: 전각·공백 정규화
# ────────────────────────────────
_ALLOW = '공영일이삼사오육칠팔구십백천만억조경'
_FULLWIDTH = str.maketrans('０１２３４５６７８９，', '0123456789,')

def _prep(txt: str) -> str:
    txt = txt.translate(_FULLWIDTH)                       # 전각 → ASCII
    txt = re.sub(rf'(?<=([{_ALLOW}]))\s+(?=[{_ALLOW}])', '', txt)  # 한글숫자 간 공백
    txt = re.sub(r'(?<=\d)\s+(?=[\d,])', '', txt)                    # 숫자 간 공백
    txt = re.sub(rf'([{_ALLOW}])\s+(?=\d)', r'\1', txt)             # 단위 뒤 공백
    return txt

# ────────────────────────────────
# 1) 한글(혼합) 숫자 → 정수
# ────────────────────────────────
_DIG = {k: i for i, k in enumerate('영일이삼사오육칠팔구')}; _DIG['공'] = 0
for d in '0123456789': _DIG[d] = int(d)

_SMALL = {'': 1, '십': 10, '백': 100, '천': 1_000}
_LARGE = [('경', 10**16), ('조', 10**12), ('억', 10**8), ('만', 10**4)]

def _parse_4(ch: str) -> int:
    if not ch: return 0
    if ch.isdigit(): return int(ch)
    tot = tmp = 0
    for c in ch:
        if c in _DIG:
            tmp = _DIG[c]
        elif c in _SMALL:
            tot += (tmp or 1) * _SMALL[c]; tmp = 0
    return tot + tmp

def _ko2int(txt: str) -> int:
    tot, rem = 0, txt
    for u, v in _LARGE:
        if u in rem:
            left, rem = rem.split(u, 1)
            tot += (_parse_4(left) if left else 1) * v
    return tot + _parse_4(rem)

# ────────────────────────────────
# 2) 문자열 → 숫자 리스트 (아라비아·한글·혼합)
# ────────────────────────────────
_ARAB = re.compile(r'\d[\d,]*')
_KOR  = re.compile(rf'[{_ALLOW}]+')
_MIX  = re.compile(rf'[0-9{_ALLOW}]+[만억조경][0-9{_ALLOW}]*')

def extract_numbers(text: str) -> list[int]:
    text = _prep(text)
    nums: list[int] = []

    # ① '81만9265' 같이 단위 포함 혼합 표현
    for m in _MIX.findall(text):
        try:
            nums.append(_ko2int(m))
            text = text.replace(m, ' ', 1)  # 중복 방지
        except Exception:
            pass

    # ② 아라비아 숫자
    nums += [int(m.replace(',', '')) for m in _ARAB.findall(text)]

    # ③ 순수 한글 숫자
    for m in _KOR.findall(text):
        try:
            nums.append(_ko2int(m))
        except Exception:
            pass
    return nums

# ────────────────────────────────
# 3) "완전 일치(±tolerance)" 리워드
# ────────────────────────────────
@REWARD_REGISTRY.register("number_reward")
def number_reward_func(pred_text,
                       gold_text,
                       tolerance=0, **kwargs) -> float:
    """
    pred / ref  : 비교할 두 문자열
    tolerance   : 절대 오차 허용 범위(0 = 완전 일치만)
    반환값      : 일치하면 1.0, 아니면 0.0
    """
    np, nr = extract_numbers(pred_text), extract_numbers(gold_text)

    # 숫자 개수 다르면 0점
    if len(np) != len(nr) or not nr:
        return 0.0

    np.sort(); nr.sort()
    for p, r in zip(np, nr):
        if abs(p - r) > tolerance:
            return 0.0
    return 1.0


