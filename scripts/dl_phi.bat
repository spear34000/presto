@echo off
cd /d C:\Users\spear\project\presto
set HF_XET_HIGH_PERFORMANCE=1
python "C:\Users\spear\AppData\Local\Temp\opencode\dl_phi.py" > models\dl_phi.log 2>&1
python -c "from huggingface_hub import HfApi; api=HfApi(); [print('QAT repo:', m.id) for m in api.list_models(search='gemma-4 qat gguf', limit=10)]" >> models\dl_phi.log 2>&1
