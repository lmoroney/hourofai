import nbformat as nbf
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# How Chatbots Really Work: Inside a Large Language Model 💬🤖
### A free Hour of AI lab by Laurence Moroney (companion to *Your First AI Model in an Hour*)

<a href="https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab2-how-chatbots-really-work/how_chatbots_really_work.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Hi, it's Laurence again. Chances are you've used a chatbot like ChatGPT, Gemini, Claude, or Copilot. You type a question, and it writes back something that sounds like a person wrote it. It can feel like there's someone in there.

Here's the secret I want you to understand by the end of this hour: **a chatbot is a next-word predictor.** At its heart, it does one thing over and over: *look at the text so far, and guess what word comes next.* That's it. Everything else is built on top of that one trick.

You already use a tiny version of this every day: the autocomplete on your phone keyboard. Today you'll open up a real (small) language model, look inside, make it write, change how creative it is, turn it into a chatbot, and then catch it making things up.

No experience needed. No accounts, API keys, or payments. Everything runs right here, for free.
""")
md("""
## How this notebook works

Same as last time, there are two kinds of code cells:

- ▶️ **JUST RUN THIS**: click the cell and press **Shift + Enter** (or the play button). You don't need to understand every line.
- ✏️ **TRY CHANGING THIS**: change the words or numbers, run it, and see what happens. You can't break anything. If things get weird, use **Runtime → Restart and run all**.

Run the cells **in order, top to bottom**.

One promise before we start: the model we're using is **tiny** compared to ChatGPT. It's about 135 million numbers, while big chatbots have hundreds of billions. So it'll be less clever, and it will sometimes say silly things. That's actually great for learning, because its mistakes are easier to see. The big ones make the *same kinds* of mistakes, just less obviously.
""")

md("""
## Part 0: Get set up

▶️ **JUST RUN THIS.** It downloads two small open language models (about 540 MB total, usually under a minute on Colab) and sets up some helper tools. Scroll past the code; you don't need to read it.
""")
code("""
# ▶️ JUST RUN THIS (it takes a minute or so; the helpers below are what we'll use)
import torch, warnings, matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging
logging.set_verbosity_error(); logging.disable_progress_bar(); warnings.filterwarnings("ignore")

tokenizer  = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
base_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M").eval()          # just predicts text
chat_model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct").eval() # trained to chat
print("Models loaded! The model has", sum(p.numel() for p in base_model.parameters()), "numbers inside it.")
""")
code("""
# ▶️ JUST RUN THIS (helper tools; no need to read them)
def next_word_probs(text, model=None, temperature=1.0):
    model = model or base_model
    ids = tokenizer(text, return_tensors="pt").input_ids
    with torch.no_grad():
        scores = model(ids).logits[0, -1]
    return torch.softmax(scores / temperature, dim=-1)

def show_next_words(text, how_many=10, temperature=1.0, model=None):
    probs = next_word_probs(text, model, temperature)
    top = probs.topk(how_many)
    words = [repr(tokenizer.decode(int(i))) for i in top.indices]
    plt.figure(figsize=(8, how_many * 0.35 + 1))
    plt.barh(words[::-1], (100 * top.values).tolist()[::-1], color="tab:blue")
    plt.xlabel("Chance of being the next word (%)")
    plt.title(f'What comes after: "{text}"' + (f"  (temperature {temperature})" if temperature != 1.0 else ""))
    plt.show()

def generate(model, text, max_words=60, temperature=0):
    inputs = tokenizer(text, return_tensors="pt")
    settings = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 \\
               else dict(do_sample=False, repetition_penalty=1.2)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_words, pad_token_id=tokenizer.eos_token_id, **settings)
    return tokenizer.decode(out[0, inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

def build_chat_prompt(message, system=None):
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": message}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def chat(message, system=None, temperature=0, max_words=60):
    reply = generate(chat_model, build_chat_prompt(message, system), max_words, temperature)
    print("🧑 You:", message)
    print("🤖 Bot:", reply, "\\n")

print("Helpers ready!")
""")

md("""
## Part 1: What's a "large language model"?

A **language model** is a program that has learned how language usually flows. It was trained by reading an enormous amount of text (websites, books, code, articles) and practising one game billions of times:

> Here's some text with the last word hidden. Guess the hidden word. Check. Adjust your numbers a tiny bit. Repeat.

If you did the first lab, that should sound familiar: it's the same guess, check, nudge loop we used to recognize clothing. The only difference is what's being guessed.

It's called **large** because of the size: huge amounts of training text, and millions to billions of internal numbers (**parameters**). Our model has 135 million. The biggest chatbots have hundreds of billions or more.

Notice what's *not* in that description: no fact database, no understanding, no checking whether things are true. It learned **what text usually looks like**. Hold on to that idea. It explains almost everything good *and* bad about chatbots.
""")

md("""
## Part 2: Tokens, or how a model reads 🔤

Models don't actually read words or letters. They chop text into chunks called **tokens**. A token might be a whole word (`cat`), part of a word (`un` + `believ` + `ably`), a space plus a word, or punctuation. Each token is then turned into a number, because (just like pictures in the last lab) computers only work with numbers.

▶️ **JUST RUN THIS** to see how a sentence gets chopped up. Each token is shown in [brackets], and a `·` means a space.
""")
code("""
# ▶️ JUST RUN THIS
def show_tokens(text):
    ids = tokenizer(text).input_ids
    pieces = [tokenizer.decode([i]).replace(" ", "·") for i in ids]
    print(" ".join(f"[{p}]" for p in pieces))
    print("Token numbers:", ids)
    print(f"{len(text)} characters became {len(ids)} tokens\\n")

show_tokens("The cat sat on the mat.")
show_tokens("Unbelievably, my cat ate 3 tacos!")
""")
md("""
✏️ **TRY CHANGING THIS:** type your own sentences. Try your name, a long word, a made-up word, slang, another language, or an emoji. Which ones get split into lots of pieces?
""")
code("""
# ✏️ TRY CHANGING THIS
show_tokens("My name is Laurence and I love pizza")
show_tokens("supercalifragilisticexpialidocious")
show_tokens("strawberry")
""")
md("""
Common words are usually one token. Rare words, names, and other languages get split into many pieces (and emoji can even get split into pieces that don't display properly).

Look at `strawberry`. The model doesn't see ten letters, it sees a couple of chunks. So if you ask a chatbot *"how many r's are in strawberry?"*, it's being asked about letters it never actually sees! Remember that; we'll come back to it.

💬 **Talk about it:** Languages with less text on the internet often get chopped into more tokens. Why might that make chatbots work worse (or cost more) for people who speak those languages?
""")

md("""
## Part 3: Guess the next word 🎯

Now the main event. Let's give the model some text and look at what it thinks comes next. It doesn't pick just one word. It gives a **chance (probability) to every token it knows**, about 49,000 of them!

First, **you** guess: what word comes after *"Once upon a time, there was a"*? Say it out loud. Then ▶️ **JUST RUN THIS.**
""")
code("""
# ▶️ JUST RUN THIS
show_next_words("Once upon a time, there was a")
""")
md("""
Did you match the model? Notice that it isn't *sure*. It spreads its bets across lots of possibilities, just like you would.

✏️ **TRY CHANGING THIS:** try your own starts. Some ideas: `"The capital of France is"`, `"My favorite food is"`, `"I can't come to school today because"`, `"The best video game ever is"`.
""")
code("""
# ✏️ TRY CHANGING THIS
show_next_words("The capital of France is")
""")
md("""
Interesting, right? For *"The capital of France is"* the top guess often isn't even "Paris". It's something like "the", because sentences like *"The capital of France is the city of Paris"* are also common. The model isn't looking up a fact; it's predicting **likely text**.

### Watch it write, one token at a time

A chatbot writes a whole answer by doing this over and over: predict a token, add it to the text, predict the next one... ▶️ **JUST RUN THIS** to watch it happen step by step.
""")
code("""
# ▶️ JUST RUN THIS
text = "The best thing about summer is"
for step in range(12):
    probs = next_word_probs(text)
    best = int(probs.argmax())
    word = tokenizer.decode(best)
    print(f"Step {step + 1:2d}: picked {word!r:14} ({100 * probs[best]:.0f}% chance)  ->  {text + word}")
    text = text + word
""")
md("""
That's the whole trick. Every chatbot answer you've ever read was built exactly like this: **one token at a time**, each one a prediction based on everything before it.
""")

md("""
## Part 4: Temperature, or how creative should it be? 🌡️

In the last cell, the model always picked the **most likely** word. That's safe, but it can get boring and repetitive. Real chatbots usually **roll the dice**: they pick words randomly, but likely words get picked more often. That's why you can ask ChatGPT the same question twice and get different answers.

A setting called **temperature** controls how adventurous the dice roll is:
- **Low temperature (like 0.2)**: the most likely word almost always wins. Predictable, focused, a bit dull.
- **Temperature 1**: the model's normal odds.
- **High temperature (like 2)**: unlikely words get a big boost. Surprising... and eventually nonsense.

▶️ **JUST RUN THIS** to see how temperature reshapes the odds for the same sentence.
""")
code("""
# ▶️ JUST RUN THIS
text = "My favorite animal is the"
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharex=True)
for ax, t in zip(axes, [0.3, 1.0, 2.0]):
    top = next_word_probs(text, temperature=t).topk(8)
    words = [repr(tokenizer.decode(int(i))) for i in top.indices]
    ax.barh(words[::-1], (100 * top.values).tolist()[::-1], color="tab:orange")
    ax.set_title(f"temperature {t}"); ax.set_xlabel("Chance (%)")
fig.suptitle(f'What comes after: "{text}"'); plt.tight_layout(); plt.show()
""")
md("""
All three charts use the same scale. At low temperature the top few words get most of the chance. At high temperature the bars nearly vanish: the chance gets spread across tens of thousands of other tokens, including weird ones, so almost anything can come out. Now let's see what that does to actual writing.

✏️ **TRY CHANGING THIS:** change `temperature` to `0.2`, then `1.0`, then `2.0` (or even `3.0`). Run each one a couple of times. What happens to the poem?
""")
code("""
# ✏️ TRY CHANGING THIS
temperature = 0.2

torch.manual_seed(1)
chat("Write a short poem about the ocean.", temperature=temperature, max_words=40)
""")
md("""
✏️ **TRY CHANGING THIS:** run this cell **three times** without changing anything. Do you get the same answer each time? Why or why not?
""")
code("""
# ✏️ TRY CHANGING THIS (run it several times; try changing the question too)
chat("Give me a fun name for a pet goldfish.", temperature=1.0, max_words=25)
""")
md("""
💬 **Talk about it:** If you wanted a chatbot to help write a creative story, would you want high or low temperature? What about for answering a maths question? Different tasks need different settings.
""")

md("""
## Part 5: How do you turn a word predictor into a chatbot? 🛠️

We've actually got **two** models loaded:

1. **The base model** was only trained to predict the next word on internet text. It's great at *continuing* text, but it has no idea it's supposed to *answer* you.
2. **The chat model** is the *same* base model, trained a bit more on thousands of example conversations (question → helpful answer). This extra step is called **instruction tuning**. Big chatbots also get feedback from people rating answers, to nudge them toward helpful and safe replies.

Let's ask them both the same thing. ▶️ **JUST RUN THIS.**
""")
code("""
# ▶️ JUST RUN THIS
question = "Give me three tips for studying for a test."
print("🧱 BASE MODEL (just continues the text):\\n", generate(base_model, question), "\\n")
print("💬 CHAT MODEL (instruction-tuned):\\n", generate(chat_model, build_chat_prompt(question)))
""")
md("""
The base model just kind of... keeps writing, as if your question were the start of some web page. The chat model has learned the *pattern* of a helpful answer.

### The secret script: chat templates

Here's something most people never see. When you chat with a bot, your message doesn't go in by itself. It gets wrapped in a hidden script with special tokens marking who's talking. There's often a **system prompt** too: hidden instructions from the company that made the chatbot, like *"You are a helpful assistant."*

▶️ **JUST RUN THIS** to see exactly what the model *really* receives.
""")
code("""
# ▶️ JUST RUN THIS
print(build_chat_prompt("What is the moon?", system="You are a friendly tutor for students."))
""")
md("""
See it? The model is still just doing next-word prediction. It's continuing a script that says *"assistant:"*, so the most likely next text is... an assistant's reply. The whole "chatbot" is really a clever **fill-in-the-script** game.

✏️ **TRY CHANGING THIS:** change the `system` instructions and see if the bot's personality changes. Try `"You are a pirate. Talk like a pirate."`, `"Answer in one short sentence."`, or `"You are a grumpy cat."`
""")
code("""
# ✏️ TRY CHANGING THIS
system = "You are a pirate. Answer every question in pirate speak, saying 'Arr!' a lot."

chat("What is the moon?", system=system)
chat("What is the moon?")  # same question with no system prompt, for comparison
""")
md("""
Honest note: our tiny model is **not very good** at following system prompts. It picks up a little of the flavour, but often ignores the instructions. Big chatbots follow them much more closely, because they're far larger and had much more training. That's a real lesson too: **how well a model follows instructions depends a lot on its size and training.**

💬 **Talk about it:** Every chatbot company writes a hidden system prompt. What kinds of instructions do you think they include? Should users be allowed to see them?
""")

md("""
## Part 6: Where chatbots go wrong ⚠️

This is the most important part of the lab. Remember: the model predicts **what text usually looks like**, not what's **true**. Let's see what that means in practice.

### 1. Hallucinations: confidently making things up

I invented a scientist for this lab. **She doesn't exist.** Let's ask the chatbot about her. ▶️ **JUST RUN THIS.**
""")
code("""
# ▶️ JUST RUN THIS
chat("Tell me about the famous scientist Dr. Maribel Quintero-Zhang and her discovery.", max_words=90)
""")
md("""
It invented a whole life story: dates, places, a university, a field of science. It sounds confident and detailed, and **every word of it is made up.** This is called a **hallucination**.

Why does it happen? Because *"Tell me about the famous scientist Dr. ___"* is usually followed by a biography. So the model writes something that **looks like** a biography. It has no built-in way to say "I've never heard of this person", unless it was specifically trained to.

It gets worse. ▶️ **JUST RUN THIS** and look closely at the sources.
""")
code("""
# ▶️ JUST RUN THIS
chat("What are three good sources about the history of the printing press? Include links.", max_words=80)
""")
md("""
Chatbots are famous for inventing **sources, quotes, and citations** that look real but don't exist. Real people (including lawyers and students!) have gotten into serious trouble for trusting them. **Always check that a source actually exists before you use it.**

### 2. Change the wording, change the answer

✏️ **TRY CHANGING THIS:** these ask the same thing in two different ways. Does the bot give the same answer both times? Try making up your own pair.
""")
code("""
# ✏️ TRY CHANGING THIS
chat("Is a tomato a fruit?")
chat("Tomatoes are vegetables, right?")
""")
md("""
Same question, different wording, and (at least when I tested it) the two answers **contradict each other**. One of them even gets the basic fact wrong: botanically, a tomato *is* a fruit. The wording changes which words are likely to come next, so it changes the answer.

Big chatbots are much more consistent than our tiny model, but they have a related habit: they often **agree with whatever you suggest** (researchers call this *sycophancy*). So if you ask "I'm right, aren't I?", a chatbot is a poor judge. Ask neutral questions, and check important answers somewhere else.

### 3. Tokens strike back: counting and maths

▶️ **JUST RUN THIS.** Remember `strawberry` from Part 2?
""")
code("""
# ▶️ JUST RUN THIS
chat("How many letter r's are in the word strawberry?", max_words=40)
chat("What is 17 times 23?", max_words=40)
""")
md("""
(The answers are **3** and **391**.) The model never sees individual letters, and it doesn't do arithmetic. It predicts text that *looks like* an answer. Big chatbots are much better at this now, often because they're connected to extra tools like a calculator, but the underlying weakness is the same.

### 4. Bias: the model learned from us

The model learned from human writing, so it also learned our **stereotypes**. Let's measure it. For each job, we'll ask the base model how likely the next word is to be **"he"** vs **"she"** after *"The ___ said that"*. ▶️ **JUST RUN THIS.**
""")
code("""
# ▶️ JUST RUN THIS
jobs = ["nurse", "babysitter", "teacher", "doctor", "engineer", "CEO", "pilot"]
he_id, she_id = tokenizer.encode(" he")[0], tokenizer.encode(" she")[0]
he, she = [], []
for job in jobs:
    probs = next_word_probs(f"The {job} said that")
    he.append(100 * probs[he_id].item()); she.append(100 * probs[she_id].item())

x = range(len(jobs))
plt.figure(figsize=(9, 3.5))
plt.bar([i - 0.2 for i in x], he, width=0.4, label='"he"')
plt.bar([i + 0.2 for i in x], she, width=0.4, label='"she"')
plt.xticks(list(x), jobs); plt.ylabel("Chance of next word (%)")
plt.title('"The ___ said that ..."'); plt.legend(); plt.show()
""")
md("""
Nobody programmed the model to think nurses are women and engineers are men. It **absorbed** those patterns from the text it read, because that's what human writing has historically looked like. Now imagine a chatbot helping to screen job applications, write news, or give advice. Those hidden patterns can quietly shape real decisions.

✏️ **TRY CHANGING THIS:** go back to the cell above and edit the `jobs` list. Try `"scientist"`, `"dancer"`, `"chef"`, `"programmer"`, `"cleaner"`. What do you notice?
""")

md("""
## Part 7: Using chatbots wisely 🧭

Chatbots can be genuinely useful. I use them myself! The trick is to use them like a **clever but unreliable study partner**, not like an encyclopedia. Here are my tips:

1. **Verify facts** using a trusted source (textbook, library database, official website). Treat a chatbot's answer as a *starting point*, not the final word.
2. **Never trust a citation you haven't found yourself.** If you can't find the source, it may not exist.
3. **Watch out for confidence.** A fluent, confident answer is *not* evidence that it's true.
4. **Ask neutral questions.** "Is X true?" is better than "X is true, right?"
5. **Use it to think, not to skip thinking.** Great: "Explain this another way", "Quiz me", "What might I be missing?" Not great: copying an answer you don't understand.
6. **Follow your school's rules, and be honest** about when and how you used AI.
7. **Protect your privacy.** Don't type personal information (yours or anyone else's) into a chatbot.
8. **Notice bias.** Ask yourself whose voices and viewpoints might be missing from an answer.

💬 **Talk about it:**
1. After today, how would you explain to a younger kid what a chatbot actually is?
2. The chatbot made up a scientist *and* sources, and sounded confident. When could that cause real harm?
3. Chatbots learn from text written by people. Whose writing is over-represented on the internet, and whose is missing? How might that affect answers?
4. Should chatbots be required to say "I don't know"? Why is that hard for a next-word predictor?
5. What's one way you'll use (or not use) chatbots differently after this lab?
""")

md("""
## 🎉 You did it!

In about an hour you:

- ✅ Learned that a chatbot is, at heart, a **next-word predictor**
- ✅ Saw how text becomes **tokens** and numbers
- ✅ Looked inside a real language model at its **next-word probabilities**
- ✅ Controlled its creativity with **temperature**
- ✅ Discovered how a word predictor becomes a **chatbot** (instruction tuning, chat templates, system prompts)
- ✅ Caught it **hallucinating**, flip-flopping, miscounting, and showing **bias**
- ✅ Learned how to use chatbots **critically and safely**

You now understand chatbots better than most adults do. Seriously! Use that knowledge well.

### 📝 Exit ticket
Answer in your own words:
1. *"A chatbot works by..."*
2. *"One way a chatbot can go wrong is... so I should always..."*

### Want to keep going?
- Try the first lab, *Your First AI Model in an Hour*, to train your own image classifier: https://colab.research.google.com/github/lmoroney/hourofai/blob/main/lab1-first-model-in-an-hour/first_model_in_an_hour.ipynb
- Invent your own fake person, book, or event and see if the chatbot makes up facts about it.
- Find a question where the base model and the chat model give *really* different answers.
- More from me: [my blog](https://laurencemoroney.com) and [my PyTorch book](https://link.amazon/B0eNBlqeN)

Thanks for learning with me. Stay curious, and stay skeptical. — Laurence
""")

nb = nbf.v4.new_notebook()
nb["cells"] = C
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"},
                  "colab": {"provenance": [], "name": "how_chatbots_really_work.ipynb"}, "accelerator": "None"}
nbf.write(nb, "/workspace/hour-of-ai-lab/llm-intro/how_chatbots_really_work.ipynb")
print("cells:", len(C), "code cells:", sum(c.cell_type == "code" for c in C))
