from tune import CPMAntNLGTune, CPMAntNLUTune
from infer import CPMAntNLGInfer, CPMAntNLUInfer, CPMAntScoreInfer

task_config = {
    "CCPM": {"tune": CPMAntNLGTune, "infer": CPMAntNLGInfer},
    "KdConv_film": {"tune": CPMAntNLGTune, "infer": CPMAntNLGInfer},
    "KdConv_travel": {"tune": CPMAntNLGTune, "infer": CPMAntNLGInfer},
    "KdConv_music": {"tune": CPMAntNLGTune, "infer": CPMAntNLGInfer},
    "Math23K": {"tune": CPMAntNLGTune, "infer": CPMAntNLGInfer},
    "C3": {"tune": CPMAntNLUTune, "infer": CPMAntNLUInfer},
    "LCSTS": {"tune": CPMAntNLGTune, "infer": CPMAntNLGInfer},
    "Sogou": {"tune": CPMAntNLUTune, "infer": CPMAntScoreInfer}
}
