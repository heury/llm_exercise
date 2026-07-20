"""
lerobot==0.5.0's (unused) groot policy has a dataclass field-ordering bug:
GR00TN15Config declares several `field(init=False)` attributes with no
default, followed by one that does have a default, which Python's
dataclass machinery rejects even though the class defines its own
__init__ (the auto-generated __init__ is still built - and validated -
before being discarded). This blocks importing anything under
lerobot.policies.* or lerobot.envs.factory, which this tutorial needs
for the pi/smolvla policies it actually uses.

Run this once after `uv sync` (site-packages edits don't survive a
resync): `uv run python scripts/patch_lerobot_groot.py`
"""
import re
from pathlib import Path

import lerobot

target = Path(lerobot.__file__).parent / "policies" / "groot" / "groot_n1.py"
text = target.read_text(encoding="utf-8")

fields_needing_default = ["backbone_cfg", "action_head_cfg", "action_horizon", "action_dim"]
patched = text
for name in fields_needing_default:
    patched = re.sub(
        rf"({re.escape(name)}: \S+ = field\(init=False,)(?! default)",
        r"\1 default=None,",
        patched,
    )

if patched == text:
    print(f"Already patched (or pattern not found): {target}")
else:
    target.write_text(patched, encoding="utf-8")
    print(f"Patched: {target}")
