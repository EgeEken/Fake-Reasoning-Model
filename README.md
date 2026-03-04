# Fake Reasoning Model Mini-Project

To gain some experience on using LoRA to finetune LLMs, i decided to make a little project to fine-tune a model to gain a new feature, this feature had to be simple enough that i could quickly train for it locally. I decided to go with the “Fake Reasoning Model” idea where the model learns to respond to every single prompt starting with a fake reasoning block saying “Hmmmm…..” as if the model is actually thinking about what to say.

Final result:

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image.png)

---

## Idea and Dataset

The idea was to have the model start each answer with “<thinking> Hmmm… </thinking>” imitating actual reasoning tokens, but with nothing inside, to keep things slightly more interesting, I wanted the length of the “Hmmm…” to be variable. So for this, i generated a new dataset by taking 200 Q/A samples from the `databricks-dolly-15k` dataset, and adding this to the start with a random amount of “m” and “.”s.

### Sample example:

<aside>
  
**`User`**: What is one year?

</aside>

<aside>
  
**`Assistant`**: \<thinking> Hmmmmmmmm..... \</thinking> 

One Earth Year is the time it takes for one revolution of the Earth around the sun.

</aside>

---

## Architecture

### Base model

I used Qwen 2.5 - 0.5B as the base model so that local inference on my work macbook would be quick and training would be possible at all.

### LoRA Architecture

I experimented with a bunch of different hyperparameters but most of them either caused crashes due to insufficient memory or just didn’t learn, after some experimentation i settled on these:

**LoRA Hyperparameters**:

- Rank $( r ) = 4$
- Alpha $(\alpha) = 2 * r = 8$
- Target Modules = `[”q_proj”, “v_proj”]`
- Dropout = $0.05$

**Training Hyperparameters**:

- Max Length = $256$
- Effective Batch Size = $4$
    - Per Step Batch Size = $2$
    - Gradient Accumulation Steps = $2$
- Learning Rate = $0.0002$
- Epochs = $3$

---

## Results and Findings

It works to do what I wanted it to do, admittedly this was a very achievable goal, but it was achieved so good news.

Funny enough, in this one the fake reasoning model performed better, even though there was actually *less* reasoning being done:

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%201.png)

And certainly the 200 sample Q&A dataset i quickly generated that didn’t even have any mathematical reasoning questions, could not have given the fine-tuned model a newfound capacity for it. I tested a few more questions and saw the same thing happen twice so i got a little curious, and started working on a benchmark of mathematical reasoning capacity to see if this somehow had an observable, positive effect or if it was just random luck. 

**NOTE**: For the purposes of the initial goal which was to get some practical experience using LoRA and the hugging face PEFT library, the project was a success it trained and worked as intended, everything else after this part is just extra work to examine an interesting phenomenon i saw

---

### Basic Arithmetic Benchmark

I made a benchmark that generates simple 2 digit arithmetic questions and parses through each model’s response to find their final answer number, comparing it to the correct pre-determined answer. I ran this benchmark for 1000 questions:

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%202.png)

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%203.png)

There are indeed some baffling outliers where the LoRA seems to just randomly perform much better despite using a lot less reasoning tokens, and as seen above on average it is slightly better.

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%204.png)

But overall, results were largely identical in terms of ability, there are certain cases where one model gives a correct answer while the other fails but there’s no statistically significant difference in accuracy, the LoRA model performs ever so slightly better in terms of accuracy, but the 0.3% difference could just be explained by randomness better than anything.

However, i noticed that in a lot of the examples, even when both models were wrong, the LoRA fine-tuned model would only be off by a little bit usually, while the base model could be very far off. And checking the stats on just how far off each response is, there is a significant distinction in average difference to the correct answer, which was surprising.

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%205.png)

This is such a massive difference that it seems like the LoRA fine-tuned model just performs better at arithmetic for no reason, almost twice as much! And this is done over 1000 test questions, same questions for both too, there’s no room left for luck that could explain a whole 2x difference.

I did some further analysis to figure out a pattern, if it’s just a few extremely far off outliers that ruin the base models:

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%206.png)

(Starting at 75 because obviously the 78% accuracy means the bottom 78 percentiles are 0)

We can see that it isn’t just the effect of a few individual outliers, the LoRA fine-tuned model generally performs better. In fact, when we cut the top 5% of worst outliers, the performance difference is actually much bigger, going up to over 7x better compared to the 2x before.

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%207.png)

---

### Subliminal Learning?

This difference kinda reminds me of a paper i read a while ago:

https://arxiv.org/html/2507.14805v1

In this paper, the researchers observed “subliminal” information/preference transfer through fine-tuning, even when the fine-tuning data itself contained no information or relevant content to those behaviors being transferred.

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%208.png)

While obviously this is a completely different finding, there is a chance the root cause is a similar kind of “subliminal” learning, although one that could be more useful, where fine-tuning on this dataset just subliminally transfers mathematical ability? More research needs to be done on other models / other datasets before there’s anything definitive that comes out of this, but still very interesting findings that i did not expect to see here when i started this project.

Hypothesis 1: It’s subliminal learning, somehow this dataset, without ever covering mathematics, teaches the model how to be better at math.

Hypothesis 2: It’s the fake thinking, by spending extra tokens that aren’t helpful, but also guaranteed to not be misleading, the model gets to redistribute its cognition inside it and do better.

---

## Further testing

My internship tutor suggested doing two things:

1. Normalizing the differences rather than getting the direct numbers
2. Fine-tuning the base model on the same dataset, without the fake thinking prefixes, as a control group so that we know if it’s the dataset or the fake thinking tokens (Hypothesis 1 or 2)

---

### 1. Normalizing

Implementing MAPE (Mean Absolute Percentage Error) to see differences in scale rather than absolute numbers. Instead of averaging the absolute differences:

$$
Difference = \frac{1}{n} \sum_{t=1}^{n} |A_t - F_t|
$$

We average the MAPE:

$$
MAPE = \frac{1}{n} \sum_{t=1}^{n} \frac{\left|A_t - F_t\right|}{\left|A_t\right|}
$$

This is similar and should keep the same logic except instead of punishing bigger numbers being off more strongly, it punishes by scale rate. Using this, we get these results:

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%209.png)

Which at first seems like it invalidates the whole difference but this is due to just a handful of extreme outliers in the LoRA results, so if we just shave off the worst 5% of answers (50 examples), we get this, again, LoRA version is about 2x closer on average in the vast majority of cases.

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%2010.png)

Even just cutting 1 single worst outlier, reducing the n=1000 to n=999, we can see a clear 20% difference between the two models:

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%2011.png)

---

### 2. Control Group LoRA

I fine-tuned a second model, on the same dataset, with the same LoRA and training hyperparameters, just with the fake thinking prefixes removed from the training dataset, to create a control group LoRA model.

Surprisingly, the control model performed even better than the previous LoRA, proving hypothesis 2 wrong, and 1 most likely right.

![Even the accuracy is slightly higher now](Fake%20Reasoning%20Model%20Mini-Project/image%2012.png)

Even the accuracy is slightly higher now

![Previous difference metric](Fake%20Reasoning%20Model%20Mini-Project/image%2013.png)

Previous difference metric

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%2014.png)

![New MAPE metric](Fake%20Reasoning%20Model%20Mini-Project/image%2015.png)

New MAPE metric

![image.png](Fake%20Reasoning%20Model%20Mini-Project/image%2016.png)

Here there is absolutely no ambiguity, fine-tuning the model on the 200 q/a sample from the dataset with no mathematics, somehow improves the mathematical capacity of the model. Which is interesting in my opinion.
