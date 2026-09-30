# Verification run: 2026-09-30 (PT)
- Environment: box (Linux, Python 3.13.5), torch 2.14.1+cpu, torchvision 0.29.1+cpu, CPU only.
- Command: `jupyter nbconvert --to notebook --execute first_model_in_an_hour.ipynb --output first_model_in_an_hour.executed.ipynb`
- Result: all 19 code cells ran with no errors (including a fresh Fashion-MNIST download).
- Total wall time: 46 s with 2 CPU threads (similar to free Colab's 2 vCPUs); 45 s with 1 thread. About 3-4 s per training epoch.
- Main model (784-128-10 MLP, Adam lr=0.001, 5 epochs, seed 42): untrained accuracy 11.7%, then 84.2 / 85.2 / 86.5 / 86.7 / **86.7%** test accuracy; loss 0.554 to 0.313.
- Experiment A (8 hidden neurons, 3 epochs): 81.0%. Experiment B (lr=0.1, 3 epochs): 32.8% (overshoot, as intended).
- Per-class accuracy: Shirt 65%, Pullover 65% (worst); Bag 97%, Sandal 96% (best). 1,334 of 10,000 test images wrong.
- "White background" demo: original ankle boot is classified Ankle boot (90%); the inverted image is classified **Bag (100% sure)**.
- Expect Colab to run slower than the box (roughly 1-3 min end to end). Not tested on Colab itself.
