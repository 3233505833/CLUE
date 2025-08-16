# CLUE

Here is the code for the LLM-based usefulness judgment method — CLUE. We used two types of LLMs.

## GPT

You can run the code of CLUE by right-clicking on  `CLUE.py`. You can switch datasets in the YAML configuration file. Additionally, we provide baseline implementations including `pointwise.py`, `pairwise.py`, and `listwise.py`, as well as machine learning method. These can also be run by right-clicking.

## LLAMA

### Step 0: Prepare Data

First, place the training and test datasets in the `zhdata` directory, and set the validation split ratio from the training set in `my.yml`.

### Step 1: Fine-Tuning

```
python trl_finetune.py -c configs/my.yml
```

### Step 2: Merge Model

```
python merge_lora.py -c configs/my.yml
```

### Step 3: Run Inference

```
python inference.py
```
## UUST DATA
### A Brief Introduction:
To better understand the real thoughts of users, we conduct a user study where participants are asked to think aloud while reporting usefulness scores, following the think-aloud protocol. Collecting and analyzing think-aloud data is a method commonly used to develop models of cognitive processes during problem-solving tasks. By this approach, real-time reflection helps us gain deeper insights into their decision-making process and better understand the factors influencing their judgments.

They are required to complete these search tasks while thinking out aloud. At the same time, we begin recording both video and audio, and a browser plugin begins logging users’ search behaviors. After completing each task, the participant is asked to annotate the usefulness of the documents they click on and provide final query-level satisfaction.

Once all tasks are completed, we organize and manually transcribe the audio recordings from key time points—when participants annotate usefulness.

After the experiment, to get the full document content, we immediately take screenshots of the web pages and perform OCR to save the content.

We design 10 tasks covering numerous categories like medical inquiries, homework assistance, and computer knowledge. We recruit 34 participants through social networks, and ultimately, data from 31 participants are retained. All 31 participants are current students (undergraduate, master’s, and doctoral levels). Among the participants, 18 are female, and 13 are male.

We recreate a traditional web search engine style, for our study.

The participant can use this experimental search engine in the same way they normally do when using commercial search engines, and each time a user submits a query, the SERP page displays ten search results retrieved from a commercial search engine in real time. We use a browser plugin to log users’ search interactions, including all clicks and movements.

### Usage Guide:

The document content includes more than just the “title” and “snippet” fields in the JSON file. It is linked to the document content in the corresponding folder within the zip attachment through the “SERP_id” and the corresponding rank. To keep the JSON concise, the document content is provided as an attachment.

This dataset supports two tasks: user simulation and usefulness prediction. The “thought” field represents the user’s thoughts when clicking. The “usefulness_thought” reflects the user’s thoughts when annotating usefulness.

“dwelltime” indicates the time a user stayed on the document. Please note that for a very small portion of documents under wenku.baidu.com and video domains, the dwell time may be shorter due to redirection. You may consider filtering out these documents.



