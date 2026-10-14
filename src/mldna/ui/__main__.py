"""`python -m mldna.ui [workspace_dir]` - the same as `mldna ui`."""
import sys

from mldna.ui.server import serve

serve(sys.argv[1] if len(sys.argv) > 1 else ".")
