"""运行四份 demo notebook，汇总结果"""
import sys, json, os
sys.path.insert(0, '.')

def run_nb(nb_path):
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    code_cells = [(i, ''.join(c['source'])) for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
    glb = {'__name__': '__main__', '__builtins__': __builtins__}
    def display(x):
        import pandas as pd
        if isinstance(x, pd.DataFrame):
            print(x.to_string(index=False))
        else:
            print(x)
    glb['display'] = display
    errors = []
    for cell_idx, src in code_cells:
        try:
            exec(compile(src, '<cell[%d]>' % cell_idx, 'exec'), glb)
        except Exception as e:
            import traceback
            traceback.print_exc()
            errors.append((cell_idx, type(e).__name__, str(e)[:120]))
    return errors

notebooks = [
    'demo_lgbm_clf.ipynb',
    'demo_xgb_clf.ipynb',
    'demo_lgbm_reg.ipynb',
    'demo_xgb_reg.ipynb',
]

all_ok = True
for nb in notebooks:
    print('\n========== %s ==========' % nb)
    errs = run_nb(nb)
    if errs:
        all_ok = False
        for cell_idx, etype, msg in errs:
            print('  FAIL cell[%d] %s: %s' % (cell_idx, etype, msg))
    else:
        print('  [OK]')

print('\n========== 汇总 ==========')
if all_ok:
    print('四份 demo 全部通过')

print('\n-- 报告文件 --')
for root, dirs, files in os.walk('./reports_demo'):
    for f in sorted(files):
        path = os.path.join(root, f)
        print(' ', path)

print('\n-- 模型文件 --')
for root, dirs, files in os.walk('./saved_models'):
    for f in sorted(files):
        path = os.path.join(root, f)
        if not any(skip in path for skip in ['test_pkl', 'test_joblib', 'lgbm_reg_custom', 'lgbm_demo']):
            print(' ', path)
