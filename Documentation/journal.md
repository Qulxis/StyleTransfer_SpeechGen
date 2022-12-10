## References:
https://github.com/fuzhenxin/Style-Transfer-in-Text
https://arxiv.org/pdf/1705.09655.pdf


## Engineering Process:
The goals of this project is to first create a metric of style and exemplify its effectiveness at differentiating between users. The second task is to bind a decoder by this metric and observe the results. At the end, I look to consider what can be learned and how this could be improved.

Abstract:
Measuring style in text is very difficult to measure even for humans. As such, it is difficult to even measure the quality of any metric. As defined in (https://arxiv.org/abs/2109.15144#:~:text=Text%20style%20transfer%20is%20defined,meaning%20of%20the%20original%20sentence) " Text style transfer is defined as a task of adapting and/or changing the stylistic manner in which a sentence is written, while preserving the meaning of the original sentence."

Previous attempts at style transfer have largely failed at this task when using sentiment transfer.
Examples of this in practice have been seen in (https://arxiv.org/pdf/1703.00955) which uses sentiment as a constraint for text generation based on a meaning vector. In other words, sustaining the same meaning while differing the sentiment in the generation. (https://github.com/shentianxiao/language-style-transfer) similarly looks to transform a sentences "style" by reversing its sentiment as well as (/https://arxiv.org/pdf/1705.09655.pdf). These methods however, are not true style transfer as the results also differ in meaning. In all cases, the results did no preserve meaning, but rather the subject and topic of the original sentence. For example, in (/https://arxiv.org/pdf/1705.09655.pdf), the sentence, "it was super dry and had a weird taste to the entire slice" transformed to "it was a great meal and the tacos were very kind of good" with sentiment transfer. However, as we can see, the meaning of the sentence was not preserved. 

Sentiment is difficult to work with as a measure of style because it also will be affected by the meaning of the sentence as well. In other words, sentiment transfer is not necessarily true style transfer. Looking at other methods of style transfer so far, in (https://arxiv.org/pdf/2109.15144.pdf), we can observe that politness, formality, offensiveness, personal and genre styles are still viable. 


To work in style in text, we must first define what style is. First, I must differentiate based on the medium. According to (https://literarydevices.net/style/), style in literature is defined as an author's syntax, word choice, and tone. They go further to define four styles of writing. These are expository, descriptive, persuasive, and narrative style. 

Main:
Previous work looks at specifying differences in style by 

Revamp: (politeness)
Make polite, formal, offensive, or personal versions of a twitter bot. A couple options: take a tweet generation, and then apply politness (based on previous metric), or select tweets that are defined to be "polite" and use that as filter to make generated tweets artificially polite (or non-polite). In other words, rather than using a linear metric of politeness, we use a metric of an individual's style of politeness being applied to text. This is very useful because we can get a lower and upper bound of what politeness is for each user. For example, user A may express their most polite texts in very curt text, which may be considered very rude in the style of another user. It is important to localize politeness and other style metric to each user.

Another option: filter out tweets by politeness and then fine tune BERT/GPT2 to create different bots of different levels of politeness.
We need a way to measure politness.

Step 1: Detect politness
- Automated (using previous work but uses R and poor documentation): https://cran.r-project.org/web/packages/politeness/vignettes/politeness.html
- https://journal.r-project.org/archive/2018/RJ-2018-079/RJ-2018-079.pdf
- Automated standford: https://github.com/sudhof/politeness
- 



Engineering Process:
1. Wanted to create twitter bot that would generate tweets in the style of a specific user, but contrained to a specific subject. Original thought was to use previous fine tuning GPT2 models on twitter users and then use prompts as a way to contrain the subject of the tweet (see previous work as to why this mighth have been viable by previous metrics, ie sentiment transfer was allowed to change meaning, as long as subject was still there)
2. After more research, I deduced firstly that I was unsatified by the previous versions of measuring meaning in style transfer (sentiment transfer changes meaning!) thus I switch to a subset of "style". 


Sources: 
https://www.philschmid.de/fine-tune-a-non-english-gpt-2-model-with-huggingface
https://github.com/fuzhenxin/Style-Transfer-in-Text

API for stanford ConvoKit.
https://convokit.cornell.edu/documentation/politenessStrategies.html
https://dl.acm.org/doi/abs/10.1145/3415190
Politness in style transfer context
https://www.cs.cornell.edu/~cristian/Politeness_Paraphrasing_files/fine-grained-politeness-paraphrasing.pdf
