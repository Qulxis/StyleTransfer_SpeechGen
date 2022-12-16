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


## Other approaches:

CVAEs:
https://arxiv.org/pdf/1511.06349.pdf
https://arxiv.org/pdf/1703.10960.pdf Section 3.1
- Unsure how a CVAE would work? We don't have a "Correct" output.
- How to condition this during training? CVAE require a context, c, which requires knowing the topic. I don't have the means to label the topics of each tweet so unsure what my context vector would be. I could extract the nouns and their order to form my context, but then I worry that isn't much better than just using a prompt. I could maybe do a train test split though and use the nouns extracted as my context vector in both cases and the "correct output" would just be the original tweet. I think this might just end up being MLM to a certain degree if I force an ordering of the nouns, and if I don't force an order, I might as well just use the LLM with prompt approach as I am doing now? Should talk to TAs and Kathy.


# Source DOCS:
Doesn't work :EARLY STOPPING: https://stackoverflow.com/questions/69087044/early-stopping-in-bert-trainer-instances

# Personalization
#### Other
https://aclanthology.org/2020.acl-main.700.pdf
"With the rising popularity of search engines in 1990s, the need for personalization in the interpretation of the query becomes obvious "http://www2.hawaii.edu/~donnab/lis610/TDWilson_Only_1999.pdf

#### Stanford Politeness Paper: https://nlp.stanford.edu/pubs/politeness.pdf
Theories of Politeness: https://files.eric.ed.gov/fulltext/EJ1126942.pdf
 Kaplan (1999) observes that “people desire to be paid respect” and politess is a the coin in communication. 

"...and are often decisive factor in whether those interactions go well or poorly"(Gyasi Obeng, 1997; Chilton, 1990; Andersson and Pearson, 1999; Rogers and LeeWong, 2003; Holmes and Stubbe, 2005).

"We show that polite Wikipedia editors are more likely to achieve high status through elections; however,
once elected, they become less polite. Similarly,on Stack Exchange, we find that users at the top of the reputation scale are less polite than those at the bottom"

Answers to online questions are often less polite than the questions themselves. However, high reputation users on stack exchange are less polite.

This paper's methods are distint to its domains and the author differentiate that in their papers:
"However,
this research focusses on domain-specific textual
cues, whereas the present work seeks to leverage domain-independent politeness cues, building on the literature on how politeness affects
worksplace social dynamics and power structures
"

#### Why we need this work:
(cite https://aclanthology.org/2020.acl-main.700.pdf)

" personalization includes also an intrauser
modeling of different individual contexts based on
user’s communication goals."

"empirical orstatistical NLP area (Manning et al., 1999; Brill and
Mooney, 1997), the focus on frequently appearing
phenomena in large textual data sets unavoidably
led to NLP tools supporting “standard English” for
generic needs of an anonymous user."

"
science argument that an identity is the product
rather than the source of linguistic and other semiotic practices, and identities are relationally constructed through several, often overlapping, aspects
of the relationship between self and other, including similarity/difference, genuineness/artifice and
authority/delegitimacy (Bucholtz and Hall, 2005 https://bucholtz.linguistics.ucsb.edu/sites/secure.lsit.ucsb.edu.ling.d7_b/files/sitefiles/research/publications/BucholtzHall2005-DiscourseStudies.pdf)

"politeness is universal as a concept but not as a behaviour"


What to do about bad language on the internet: https://aclanthology.org/N13-1037/

Word choices are often demographic and associate with different demographich group. General politeness labeling biases towards the group that it was trained on. Issues
"many of the features
that characterize bad language have strong associations with specific social variables. In some cases,
these associations mirror linguistic variables known
from speech — such as geographically-associated
lexical items like hella, or transcriptions of phonological variables like “g-dropping” (Eisenstein et al.,
2010). But in other cases, apparently new lexical
items, such as the abbreviations ctfu, lls, and af,
acquire surprisingly strong associations with geographical areas and demographic groups (Eisenstein
et al., 2011)."


It occurs often as a symbol of solidarity and indentity.

"
The use of non-standard language is often seen
as a form of identity work, signaling authenticity, solidarity, or resistance to norms imposed from
above (Bucholtz and Hall, 2005). 
"

#### Why Twitter:
What to do about bad language on the internet: https://aclanthology.org/N13-1037/
"In contrast, Twitter users in the USA contain an equal proportion of men and women, and
a higher proportion of young adults and minorities
than in the population as a whole (Smith and Brewer,
2012). Such demographic differences are very likely
to lead to differences in language (Green, 2002;
Labov, 2001; Eckert and McConnell-Ginet, 2003).


https://jannisandroutsopoulos.files.wordpress.com/2011/11/language-change-and-digital-media-preprint.pdf
"Twitter itself is not a unified genre, it is composed of many
different styles and registers, with widely varying
expectations for the degree of standardness and dimensions of variation (Androutsopoulos, 2011)
"


### Ideas:
Politeness is like payment and determines how interactions go (Stanford). However, what is polite varies from person to person. It is important to move outside of "General Language" (cite https://aclanthology.org/2020.acl-main.700.pdf). In online elections, there is a clear differentiation between elected admins and their failed counterparts: (Stanford). 



Joe Rogan: Even the rude versions aren't neccessarily rude, but they differ from generic politeness.

Interesting"

CASE FOR WHY WE NEED TO BE PERSONALIZED: JOE ROGAN IS A GOOD EXAMPLE OF THIS DEMONSTRATED.
https://aclanthology.org/P15-1073/
https://aclanthology.org/W17-1606/
Neglecting the variety of users and use cases doesn’t
make the tools universally applicable with the same
performance - it only makes our community blind
to the built-in bias towards the specifics of user profiles in training data (Hovy, 2015; Tatman, 2017).

As people disagree what is polite in the middle ground, we use the extremes of polite and impolitess scores.
cite (. As shown in Table 2, full agreement is much more common in the 1st (bottom) and 4th (top) quartiles than in the middle quartiles. Stanford)

Classifier is p good, but partly because humans aren't: human acc is 80.89%- 86.72% and bot is 75.43%-83.79%.

Politeness is a style metric used in style transfer. However, politeness as a metric from a general or "standard english" perspective is ignorant of the diversity of langauge. Transfering politeness first requires us to understand what politeness is and how by using a standard politeness approach, we miss a lot of the neuances of language and culture. 

The approach to style transfer by isolating topics such as politeness and sentiment are dangerous as by generalizing, one only reinforces the biases in the training set. 