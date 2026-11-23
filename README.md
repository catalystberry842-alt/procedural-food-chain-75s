# 5-Stage Procedural Food Chain Animation (`food_chain_75s.mp4`)

A pure Python procedural video and audio generation pipeline that synthesizes an exact **75.0-second** (`1800` frames @ `24 FPS`, `1280×720`) hand-drawn sketch animation and synchronized synth-foley soundtrack of a 5-stage biological food chain:

**Grass → Grasshopper → Frog → Snake → Hawk**

Everything — visuals (3-pass Gaussian line-boil sketch renderer, Bezier splines, L-systems, easing kinematics) and audio (NumPy/SciPy procedural sound synthesis) — is generated purely from code with zero external media assets.

## Direct Download

- **Direct MP4 Download (37.6 MB):** [Download `food_chain_75s.mp4`](https://github.com/catalystberry842-alt/procedural-food-chain-75s/raw/main/food_chain_75s.mp4)
- **GitHub File Page:** [`food_chain_75s.mp4`](https://github.com/catalystberry842-alt/procedural-food-chain-75s/blob/main/food_chain_75s.mp4)
- **Python Pipeline Script:** [`food_chain_pipeline.py`](https://github.com/catalystberry842-alt/procedural-food-chain-75s/blob/main/food_chain_pipeline.py)

## Usage

```bash
pip install -r requirements.txt
python3 food_chain_pipeline.py
```

Outputs `food_chain_75s.mp4` (`1280×720`, 24 FPS, 75.0s, H.264 + 44.1 kHz stereo AAC).
