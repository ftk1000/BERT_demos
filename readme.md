2020.02.02

# Pipelines
[huggingface quick-tour-of-pipelines](https://github.com/huggingface/transformers#quick-tour-of-pipelines)<br>
[hanxiao/bert-as-service](https://github.com/hanxiao/bert-as-service)<br>
[]()<br>
[]()<br>
[]()<br>

# Abhishek Thakur
- [2020.03: Text Extraction From a Corpus Using BERT (AKA Question Answering)](https://www.youtube.com/watch?v=XaQ0CBlQ4cY)<br>
- [2019: Approaching (almost) Any Machine Learning Problem | by Abhishek Thakur | Kaggle Days Dubai | Kaggle](https://www.youtube.com/watch?v=uWVR_axaVwk)<br>
- []()<br>
- []()<br>
- []()<br>

# ChrisMcCormickAI: BERT Research ([Buy all notebooks](https://www.chrismccormick.ai/offers/cEGFXP2Z/checkout))
<details>
  <summary> Ep.1 Key Concepts & Sources (2019.11)<br>    </summary>
    - https://www.youtube.com/watch?v=FKlPCK1uFrc&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6 <br>
    - BERT <- Transformer <- [LSTM w/ Attention] <- [Encoder/Decoder + Bi-LSTM] <- [RNN + LSTM]<br>
    - Bogus tasks: (1) Masked Language Model, (2) Next Sentence Prediction
</details>

<details>
  <summary> Ep.2 WordPiece Embeddings (2019.11)<br>    </summary>
    - https://www.youtube.com/watch?v=zJW57aCBCTk&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=2 <br>
    - BERT vocab sz = 30k, words are embedded into R^768<br>
    - Instead of working with sequence of words BERT works with sequence of embeddings.<br>
    - individual values inside feature vectors do not matter - relative proximity of vectors matters<br>
    - word piece model: (1) "embedding" = "em" "##bed" "##ding". BERT get 3 tokens (subwords) out of 1 word, (2)'kroxldyphivc' -> k-##ro-##x-##ld-##yp-##hi-##vc,  (3) bedding -> bed-##ding<br>
    - Vocab: 1=[PAD], 101=[UNK], 102=[CLS], 103=[SEP], 104=[MASK], 1000=!, 1001=\", 7607=##mus, 7613=1873<br>
</details>   
    
<details>
  <summary> Ep.3 Fine Tuning p.1 (2019.12): Prepare CoLA (The Corpus of Linguistic Acceptability) data for taxt classification with BERT    
    <br>    </summary>
    - https://www.youtube.com/watch?v=zJW57aCBCTk&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=3 <br>
    - Pros-0 of Fine Tuning: (1) Quick Development, (2) Less Data needed, (3) Better results <br>
    - Pros-1: Good for: Text Classification, Named Entoty Recognition (NER), POS Tagging, Question + Answering part of text <br>
    - Con-0: Not good for: Language modeling, Translation, Text generation<br>
    - Con-1: BERT is a very large model: 109M weights = 109*10^6*4(Bytes)/1024/1024 = 416 MBytes (Embedding layer: 24M weights, 12 layers of Transformers=12*7M=84M)<br>
    - Con-2: Slow to train, eg 10k sentence classifications with 4 epochs on GPU at colab takes ~ 10 mins <br>
    - Con-3: Slow inferencing - challenge for production deployment <br>
    - Con-4: Domain Specific Language <br>
    - code is from https://github.com/huggingface/transformers/blob/6695450a23545bc9d5416f39ab39609c7811c653/examples/text-classification/README.md <br>
    - TF2   code: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_tf_glue.py<br>
    - Torch code: https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py<br>
    - CoLA DATA SET: https://nyu-mll.github.io/CoLA/<br>
    - Sec 3: TOKENIZE, FORMATTING (Special Tokens, Attention Mask), SEQUENCES->IDs, PADDING_n_TRUNCATING, ATTENTION_MASKS, 
  TRAIN_VALIDATION_SPLIT, CONVERT_2_PyTorch_DataType<br>
    - The ATTENTION_MASKS simply makes it explicit which tokens are actual words versus which are padding.<br>
</details>
    
<details>
  <summary> Ep.3 Fine Tuning p.2 (2019.12.19): Train Text Calssificatoin Model - Use BERTForSequenceClassification class from https://github.com/huggingface or https://huggingface.co/
    <br>    </summary>
    - VIDEO: https://www.youtube.com/watch?v=Hnvb9b7a_Ps <br>
    - CODE: https://colab.research.google.com/github/ftk1000/BERT_demos/blob/master/BERT_Fine_Tuning_Sentence_Classification.ipynb#scrollTo=GLs72DuMODJO<br>
    - Sec 3: TOKENIZE, FORMATTING (Special Tokens, Attention Mask), SEQUENCES->IDs, PADDING_n_TRUNCATING, ATTENTION_MASKS, TRAIN_VALIDATION_SPLIT, CONVERT_2_PyTorch_DataType<br>
    - Sec 4: BERTForSequenceClassification, OPTIMIZER n LearningRateScheduler, TrainingLoop<br>
    - model.train(), model.eval() sets flags<br>
    - model(..., labels) outputs LOSS, model(...) w/o labels outputs logits<br>
    - model() always returns a tuple [] : loss = output[0] <br>
    - loss.backward()   : calculates gradients <br>
    - model(), opimizer(), and scheduler() are referencing each other : during the initialization of optimizer() we pass to it model.parameters, etc <br>
    - perf metric for this task = Mathew correlation coef [-1,1]. It is used because the original data set is unbalanced.<br> 
</details>
    
<details>
  <summary> Blog: http://mccormickml.com/2019/11/11/bert <br>    </summary>
    - http://mccormickml.com/2019/11/11/bert-research-ep-1-key-concepts-and-sources/ <br>
    - BERT was trained on two “fake tasks”: “Masked Word Prediction” and “Next Sentence Prediction”. <br>
    - As a byproduct of learning to do these tasks, it develops the ability to make sense of human language. <br>
    - Once the training is finished, we strip off the final layer(s) of the model which were specific to the fake task, and then apply BERT to the tasks that we actually care about. <br>
    - coming up with these two “fake tasks” is the real innovation of BERT–otherwise it’s just a large stack of Transformers, which had already been around.<br>
    - another key contribution might be Google researchers having the text data, compute resources, and audacity to train such a huge model<br>
</details>
    


# BERT (Oct 2018) is good for: 
- [2019.11 ChrisMcCormickAI: BERT Research - Ep. 1 - Key Concepts & Sources](https://www.youtube.com/watch?v=FKlPCK1uFrc&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6)<br>
- [2019.11: ChrisMcCormickAI BERT Research - Ep. 2 - WordPiece Embeddings](https://www.youtube.com/watch?v=zJW57aCBCTk&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=2)<br>
- [2019.12: ChrisMcCormickAI BERT Research - Ep. 3 : Fine Tuning - p.1](https://www.youtube.com/watch?v=x66kkDnbzi4&list=PLam9sigHPGwOBuH4_4fr-XvDbe5uneaf6&index=3)<br>
- [2020.01: BERT Document Classification Tutorial with Code](https://www.youtube.com/watch?v=_eSGWNqKeeY)<br>
[]()<br>
[]()<br>

    
- [Jay Alammar: The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning](http://jalammar.github.io/illustrated-bert/)<br>
[google search results](https://www.google.com/search?q=BERT+and+other+transformers&rlz=1C1GCEA_enUS800US800&oq=bert&aqs=chrome.2.69i57j0j69i59j46l2j69i64l3.5639j0j7&sourceid=chrome&ie=UTF-8)<br>
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova](https://arxiv.org/abs/1810.04805)<br>
- [The Annotated Transformer](https://nlp.seas.harvard.edu/2018/04/03/attention.html)<br>
- [2019: Kaggle Coffee Chat: Jacob Devlin (Google Researcher, BERT author)](https://www.youtube.com/watch?v=u91645MFytY)<br>


[]()<br>
[]()<br>
[]()<br>

# Summarization
[EVALUATION MEASURES FOR TEXT SUMMARIZATION, J.Steinberger et al, Computing and Informatics, Vol. 28, 2009, 1001–1026, V 2009-Mar-2 ](http://www.cai.sk/ojs/index.php/cai/article/viewFile/37/24)<br>
[2017: A DEEP REINFORCED MODEL FOR ABSTRACTIVE SUMMARIZATION Romain Paulus, Caiming Xiong & Richard Socher](https://arxiv.org/pdf/1705.04304.pdf)<br> 
          - HAS REFERENCES TO GOOD DATASETS
[]()<br>
[]()<br>
[]()<br>
[]()<br>
[]()<br>
[]()<br>

# Summarization Metrics

* [Bleu measures precision: how much the words (and/or n-grams) in the machine generated summaries appeared in the human reference summaries.](https://stackoverflow.com/questions/38045290/text-summarization-evaluation-bleu-vs-rouge)
   - [Bleu: a Method for Automatic Evaluation of Machine Translation](http://www1.cs.columbia.edu/nlp/sgd/bleu.pdf)

* Rouge measures recall: how much the words (and/or n-grams) in the human reference summaries appeared in the machine generated summaries.
  - [How Rogue works](http://text-analytics101.rxnlp.com/2017/01/how-rouge-works-for-evaluation-of.html)
* F1 measure to make the metrics work together: F1 = 2  (Bleu  Rouge) / (Bleu + Rouge)

## [Precision and Recall](https://en.wikipedia.org/wiki/Precision_and_recall)
   - $$\displaystyle  recall = \frac{tp}{tp + fn}  = \frac{relevant\quad AND\quad retrieved}{relevant}$$
   - $$\displaystyle  precision = \frac{tp}{tp + fp} = \frac{relevant\quad AND\quad retrieved}{retrieved}$$


