"""Flood mapping from satellite radar: how well do methods transfer between floods?

Every entry point (notebook, scripts, Streamlit app) imports from this
package, so preprocessing and evaluation can't drift apart.

Process rule: LightGBM (`pixel_model`) and PyTorch (`unet`) are never loaded
into the same Python process. On macOS each ships its own OpenMP runtime;
with both loaded, LightGBM segfaults if PyTorch came first (verified: exit
139), and PyTorch deadlocks in an OpenMP barrier if LightGBM came first
(verified with a stack sample). So `data` and `evaluation` import neither,
and the scripts run the LightGBM and U-Net stages as separate processes.
"""
