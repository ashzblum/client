#!/usr/bin/env python
"""Base case - main process init/finish.

---
run:
  - state: finished
    config: {}
    summary:
      - m1: 1
        m2: 2
"""

import wandb

wandb.init()
wandb.log(dict(m1=1))
wandb.log(dict(m2=2))
wandb.finish()