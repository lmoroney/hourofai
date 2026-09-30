# Your First AI Model in an Hour: Teacher Guide
**A free Hour of AI lab by Laurence Moroney · CS Education Week, Dec 8-14, 2026**

> **Student notebook:** https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab1-first-model-in-an-hour/first_model_in_an_hour.ipynb · **Lab home / downloads:** [HOSTING_PAGE_PLACEHOLDER]  
> **Audience:** ages 13-18, no ML background, little or no Python · **Time:** 45-60 min · **Cost:** free

Thanks for bringing AI into your classroom. You don't need to be an AI expert to run this lab. If you can open a Google Doc, you can run it. Students train a real image classifier on photos of clothing, test it, tweak it, and then (the most important bit) find out where it fails and why.

## Learning goals
By the end, students can:

1. Explain in their own words that a **model learns rules from examples** instead of being hand-programmed.
2. Describe **training** as repeated guess, check, and adjust, and read a **loss** curve.
3. Tell **training data** apart from **test data**, and explain why we test on unseen examples.
4. Run a simple **experiment** (change one variable, predict, compare).
5. Identify **limits and bias**: uneven accuracy across groups, overconfidence, and failure on data unlike the training set.

## Setup (5 minutes, before class)
- **No installs.** Everything runs in the browser on free Google Colab (CPU is fine, no GPU needed).
- Students need a Google account (a school Workspace account works if Colab is enabled by your admin).
- Open https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab1-first-model-in-an-hour/first_model_in_an_hour.ipynb → **File → Save a copy in Drive** so each student has their own copy.
- Do a dry run yourself: **Runtime → Run all**. It completes in about 1-2 minutes.
- Tip: project the notebook and do Parts 0-2 together, then let students go at their own pace.

## Timing plan
| Part | What happens | Minutes |
|---|---|---|
| Hook + setup (Part 0) | "Could you write rules to tell a sneaker from a sandal?" Open notebook, run first cell | 5 |
| 1. What is a model? | data + answers → rules; a model is a pile of numbers that gets nudged | 5 |
| 2. Meet the data | View Fashion-MNIST images; pictures are grids of numbers | 7 |
| 3. Build a tiny network | 3-layer network, ~100K parameters; untrained accuracy ≈ 10% | 5 |
| 4. Train it | 5 epochs (~1 min on Colab CPU); watch loss fall, accuracy hit ~85-87% | 8 |
| 5. Test it | Guesses on unseen images, with confidence | 5 |
| 6. Your turn | Exp. A: hidden layer size. Exp. B: learning rate (~20-40 s per run) | 10 |
| 7. Where AI goes wrong | Mistakes, per-class accuracy, "white background" failure, discussion | 10 |
| Wrap-up | Exit ticket: "An AI model is… and it can go wrong when…" | 3-5 |

**Short on time (45 min)?** Do only Experiment A in Part 6 and pick two discussion questions. **Never skip Part 7.**

## Discussion questions
1. How is teaching a computer with examples different from giving it rules? When might each approach be better?
2. The model was ~87% accurate overall but only ~65% on shirts and pullovers. Why might an overall score be misleading?
3. The model said a boot on a white background was a **bag, with near 100% confidence**. What does this tell us about how much to trust an AI's confidence?
4. Who made this dataset, and what's missing? (It's product photos from one European online retailer.) What clothing from your life or culture isn't represented?
5. Swap clothing for faces, medical scans, or job applications. What could go wrong if a model worked better for some groups than others? Who should be responsible?
6. What would you need to change to make this model work on photos from your phone?

## Common hiccups and fixes
| Symptom | Fix |
|---|---|
| `NameError: ... is not defined` | A cell was skipped. **Runtime → Run all** (or run cells top to bottom). |
| Nothing happens / cell shows `[*]` | It's still running (training takes ~1 min). Wait for the ✓. |
| "You are not the owner" / can't save | **File → Save a copy in Drive** first. |
| Download of data fails | Temporary network issue. Rerun the cell. Check school firewall allows Colab and dataset downloads. |
| Colab asks about a GPU or warns about resources | Ignore it. This lab is designed for the free CPU. |
| Learning rate 0.1 gives ~30% accuracy | That's the point! Big nudges overshoot. Great discussion moment. |
| Numbers differ slightly from a neighbor's | Normal. Small differences from randomness and hardware. |
| Runtime disconnected after idling | **Runtime → Run all** to catch back up (~1-2 min). |

## Extensions for fast finishers
- Train for 10 or 15 epochs. Does test accuracy keep improving, or level off? (Intro to **overfitting**.)
- Add a second hidden layer: `nn.Linear(128, 64), nn.ReLU()` and change the last layer to `nn.Linear(64, 10)`.
- Find the single most confident *mistake* in the test set. Would a human get it right?
- Research **convolutional neural networks (CNNs)** and why they're better for images.
- Design a fairer dataset: what would you collect, from whom, and how would you check it?

## Alignment to AI literacy themes
- **What AI is and how it works:** learning from data, parameters, training and testing (Parts 1-5).
- **Humans shape AI:** people choose the data, model size, and settings (Parts 2, 6).
- **Evaluating AI:** accuracy, per-group performance, confidence vs. correctness (Parts 5, 7).
- **Bias, limits, and societal impact:** dataset representation, distribution shift, high-stakes uses (Part 7).
- Maps well to AI4K12's "Five Big Ideas" (especially *Learning* and *Societal Impact*) and to CSTA data and analysis / impacts of computing standards. [ALIGNMENT_DETAILS_PLACEHOLDER: confirm specific standard codes before publishing]

*Questions or feedback? [CONTACT_PLACEHOLDER]*. I'd love to hear how it went in your classroom. — Laurence
