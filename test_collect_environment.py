import faulthandler, os, sys, subprocess, torch
faulthandler.enable(all_threads=True)
print("Python:", sys.version.replace('\\n',' '))
print("Torch:", torch.__version__)
print("Torch build config:")
try:
    import torch.__config__ as tc
    tc.show()
except Exception:
    print("  (torch.__config__.show() failed)")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    try:
        print("cuDNN:", torch.backends.cudnn.version())
    except Exception:
        pass
print("Environment PATH and LD_LIBRARY_PATH excerpts:")
print(" PATH=", os.environ.get("PATH","")[:1000])
print(" PYTORCH_JIT=", os.environ.get("PYTORCH_JIT", "<unset>"))
# Replace the next lines with the snippet that reproducibly crashes your process.
# Example placeholder:
# from my_module import run_my_workload
# run_my_workload()