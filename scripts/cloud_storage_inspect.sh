#!/usr/bin/env bash
set -u

echo "HOST=$(hostname)"
df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1 || true
df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1 || true

for path in \
    /home/jovyan/Efficient-Training \
    /home/jovyan/xandi281/Efficient-Training \
    /workspace-SR006.nfs2/Efficient-Training \
    /workspace-SR006.nfs2/xandi281/Efficient-Training \
    /workspace-SR006.nfs3/Efficient-Training \
    /workspace-SR006.nfs3/xandi281/Efficient-Training; do
    if [ -d "$path" ]; then
        echo "FOUND=$path"
        git -C "$path" rev-parse HEAD 2>/dev/null || true
        git -C "$path" status --short --branch 2>/dev/null | head -40 || true
        sha256sum \
            "$path/src/main.py" \
            "$path/src/optim/base.py" \
            "$path/src/optim/utils.py" \
            "$path/src/optim/memory_efficient/galore/adamw.py" \
            2>/dev/null || true
    fi
done

for root in /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
    echo "SEARCH=$root"
    find "$root" -maxdepth 4 -type d -name Efficient-Training -print 2>/dev/null | head -50
done
