from tune import CPMAntPlusNLGTune, CPMAntPlusNLUTune
from infer import CPMAntPlusNLGInfer, CPMAntPlusNLUInfer, CPMAntPlusScoreInfer

task_config = {
    "CCPM": {"tune": CPMAntPlusNLGTune, "infer": CPMAntPlusNLGInfer},
    "KdConv_film": {"tune": CPMAntPlusNLGTune, "infer": CPMAntPlusNLGInfer},
    "KdConv_travel": {"tune": CPMAntPlusNLGTune, "infer": CPMAntPlusNLGInfer},
    "KdConv_music": {"tune": CPMAntPlusNLGTune, "infer": CPMAntPlusNLGInfer},
    "Math23K": {"tune": CPMAntPlusNLGTune, "infer": CPMAntPlusNLGInfer},
    "C3": {"tune": CPMAntPlusNLUTune, "infer": CPMAntPlusNLUInfer},
    "LCSTS": {"tune": CPMAntPlusNLGTune, "infer": CPMAntPlusNLGInfer},
    "Sogou": {"tune": CPMAntPlusNLUTune, "infer": CPMAntPlusScoreInfer}
}
