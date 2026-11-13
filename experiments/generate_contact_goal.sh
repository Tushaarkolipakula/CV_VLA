#!/usr/bin/env bash
# Full Goal-suite generation, one task at a time. Original data is read-only.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
SOURCE=${SOURCE:-/home/tushaar/data/LIBERO-datasets/libero_goal}
OUTPUT=${OUTPUT:-$ROOT/data/contact_goal_v1/libero_goal}
mkdir -p "$OUTPUT"
export MUJOCO_GL=egl
export LIBERO_CONFIG_PATH="$ROOT/configs/libero_standard"
export OMP_NUM_THREADS=2
TASKS=(
  'open_the_middle_drawer_of_the_cabinet:wooden_cabinet_1'
  'open_the_top_drawer_and_put_the_bowl_inside:wooden_cabinet_1'
  'push_the_plate_to_the_front_of_the_stove:plate_1'
  'put_the_bowl_on_the_plate:akita_black_bowl_1'
  'put_the_bowl_on_the_stove:akita_black_bowl_1'
  'put_the_bowl_on_top_of_the_cabinet:akita_black_bowl_1'
  'put_the_cream_cheese_in_the_bowl:cream_cheese_1'
  'put_the_wine_bottle_on_the_rack:wine_bottle_1'
  'put_the_wine_bottle_on_top_of_the_cabinet:wine_bottle_1'
  'turn_on_the_stove:flat_stove_1'
)
for entry in "${TASKS[@]}"; do
  task=${entry%%:*};object=${entry#*:}
  if [[ -f "$OUTPUT/${task}_demo.hdf5" ]]; then
    echo "Existing output skipped; inspect its generation report: $task"
    continue
  fi
  "$ROOT/robo_env/bin/python" -u -m smolvla_pipeline.generate \
    --task-file "$SOURCE/${task}_demo.hdf5" --target-object "$object" \
    --output-dir "$OUTPUT" --levels "${LEVELS:-5}" --repeats "${REPEATS:-2}" \
    --pos-max "${POS_MAX:-0.03}" --rot-max "${ROT_MAX:-0.15}" --seed "${SEED:-0}" \
    --demos "${DEMOS:-0}" 2>&1 | tee "$OUTPUT/${task}.log"
done
