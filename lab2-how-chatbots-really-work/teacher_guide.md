# How Chatbots Really Work: Teacher Guide
**A free Hour of AI lab by Laurence Moroney · CS Education Week, Dec 8-14, 2026 · Companion to *Your First AI Model in an Hour***

> **Student notebook:** https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab2-how-chatbots-really-work/how_chatbots_really_work.ipynb · **Lab home / downloads:** [HOSTING_PAGE_PLACEHOLDER]  
> **Audience:** ages 13-18, no ML background, little or no Python · **Time:** 45-60 min · **Cost:** free, no API keys or accounts beyond Google

Most students meet AI through chatbots. This lab lifts the lid: students load a real (small, open) language model in the browser, watch it predict the next word, change its creativity, see how it's turned into a chatbot, and then catch it making things up. Lab 1 is helpful but not required.

## Learning goals
By the end, students can:

1. Explain that a chatbot is, at its core, a **next-word (token) predictor** trained on lots of human text.
2. Describe **tokens** and why models don't "see" letters.
3. Explain how **temperature** and random sampling make answers vary.
4. Describe how a base model becomes a chatbot: **instruction tuning**, **chat templates**, and **system prompts**.
5. Identify and explain **hallucinations**, **sensitivity to wording**, and **bias**, and apply **verification habits** when using chatbots for school.

## Setup (5 minutes, before class)
- **No installs, no API keys, no paid services.** Runs on free Google Colab CPU (no GPU needed).
- Part 0 downloads two small open models (**SmolLM2-135M** and **SmolLM2-135M-Instruct** from Hugging Face, ~540 MB total). On Colab this usually takes under a minute. **Tip:** have students run Part 0 as soon as they sit down.
- If a whole class downloads at once on a slow school network, stagger starts or run Part 0 while you introduce the hook.
- Open https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab2-how-chatbots-really-work/how_chatbots_really_work.ipynb → **File → Save a copy in Drive**. Do a dry run: **Runtime → Run all** (about 1-3 minutes).
- The model is **tiny** (135M parameters vs hundreds of billions for big chatbots). It's less capable and its mistakes are more obvious. That's intentional, and the notebook says so.

## Timing plan
| Part | What happens | Min |
|---|---|---|
| Hook + setup (Part 0) | "A chatbot is a next-word predictor", like phone autocomplete. Start the model download | 5 |
| 1. What's an LLM? | Trained by guessing hidden words, billions of times; predicts *likely* text, not *true* text | 4 |
| 2. Tokens | See text split into tokens and numbers; try names, slang, emoji, "strawberry" | 6 |
| 3. Next word | Guess first, then see the model's probabilities; watch it write token by token | 8 |
| 4. Temperature | Compare probability charts; poems at 0.2 / 1.0 / 2.0; same question, different answers | 8 |
| 5. Making a chatbot | Base vs chat model; the hidden chat template; system prompt experiment | 8 |
| 6. Where chatbots go wrong | Invented scientist, fake sources, contradicting answers, counting/maths, he/she bias chart | 12 |
| 7. Using chatbots wisely | Tips, discussion, exit ticket | 5-8 |

**Short on time (45 min)?** Skip the goldfish cell in Part 4 and the system-prompt experiment in Part 5. **Never skip Parts 6-7.**

## What students should see (from our test run)
- "Once upon a time, there was a" gives: little 12%, curious 10%, wise 10%.
- "The capital of France is" gives "the" (27%) ahead of "Paris" (8%). It predicts likely text, not facts.
- **Invented scientist** "Dr. Maribel Quintero-Zhang": a full fake biography (born 1934 in San Francisco, a Berkeley PhD, quantum mechanics).
- **Fake sources:** invented titles and historians ("Dr. John Smith and Dr. Michael Thompson").
- **Tomato:** the neutral question and the leading question get contradicting answers.
- **Strawberry / 17×23:** wrong answers (the correct answers are 3 and 391).
- **Bias chart:** "he" is more likely for doctor, engineer, CEO, and pilot; "she" for nurse and babysitter.
- Answers with temperature 0 should match the above. Sampled answers (temperature > 0) will vary, which is the point.

## Discussion questions
1. How would you explain to a younger kid what a chatbot actually is?
2. The bot invented a scientist and sources and sounded confident. When could that cause real harm (homework, health, news, law)?
3. Why is "I don't know" hard for a next-word predictor?
4. Whose writing is over-represented in training data, and whose is missing? How could that shape answers? (Also: languages that get split into more tokens.)
5. Companies write hidden system prompts. What might they include? Should users see them?
6. What's a good use and a bad use of a chatbot for schoolwork?

## Common hiccups and fixes
| Symptom | Fix |
|---|---|
| Part 0 is slow or stuck | It's downloading ~540 MB. Wait; rerun if it errors. Check the firewall allows huggingface.co. |
| Warning: "unauthenticated requests to the HF Hub" | Harmless. Ignore it. No token is needed. |
| `NameError: ... is not defined` | A cell was skipped. **Runtime → Run all.** |
| Bot's answer stops mid-sentence | Replies are capped (`max_words`) to keep things fast. Raise the number if you like. |
| "My answer is different from my neighbour's" | Expected when temperature > 0 (random sampling). |
| Bot ignores the system prompt | Tiny models follow instructions poorly. It's a teaching point (see Part 5 note). |
| Odd or inappropriate text at high temperature | High temperature produces random text. Lower it. Remind students it's a probability machine, not a person. |
| Runtime disconnected | **Runtime → Run all** (the models re-download). |

## Extensions for fast finishers
- Invent a fake person, book, or event and test whether the bot makes up facts. Can any wording get it to say "I don't know"?
- Find prompts where the base and chat models differ the most.
- Add jobs to the bias chart. Try other patterns such as `"The ___ was very"`.
- Build a "neutral vs leading" pair of questions that flips the answer.
- Try Lab 1 (*Your First AI Model in an Hour*) to train an image classifier: https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab1-first-model-in-an-hour/first_model_in_an_hour.ipynb

## Alignment to AI literacy themes
- **How AI works:** training on data, tokens, prediction, probabilities, sampling (Parts 1-4).
- **Humans shape AI:** instruction tuning, human feedback, system prompts, choice of data (Part 5).
- **Evaluating AI output:** hallucination, inconsistency, confidence vs correctness, verification (Part 6).
- **Bias and societal impact:** stereotypes absorbed from data, uneven support across languages (Parts 2, 6).
- **Responsible use:** academic honesty, privacy, source checking (Part 7).
- Maps well to AI4K12's "Five Big Ideas" (*Representation & Reasoning*, *Learning*, *Natural Interaction*, *Societal Impact*) and CSTA impacts-of-computing standards. [ALIGNMENT_DETAILS_PLACEHOLDER: confirm specific standard codes before publishing]

*Note on content:* output from the small model is unfiltered. It behaved appropriately in testing, but at high temperature or with student-invented prompts it can produce odd text. Preview the notebook before class.

*Questions or feedback? [CONTACT_PLACEHOLDER]* — Laurence
