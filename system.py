import matplotlib
import sys
p = [k for k in sys.modules if k.startswith("matplotlib")]
print(p)
