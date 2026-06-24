# run_linear.py - 선형(Linear) 어텐션 모델 실행 런처
# 이 스크립트는 표준 model_minimind 모듈을 선형 어텐션 변형(model_minimind_linear)으로
# 몽키패치(monkey-patch)한 후, 지정된 대상 스크립트를 실행합니다.
# 사용법: python run_linear.py <대상_스크립트.py> [인자...]
import sys, os, importlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.modules['model.model_minimind'] = importlib.import_module('model.model_minimind_linear')
target = os.path.abspath(sys.argv.pop(1))
os.chdir(os.path.dirname(target))
__file__ = target
exec(open(target).read())
