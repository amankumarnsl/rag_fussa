# 3 Target

### Cost Evaluation

### All endpoint working flow

### Improvement in system along with cost optimization

# ------------------------------

## optimized way time calculation
```
The Real Bottlenecks (in order):
Pinecone Search: 13.5s (52% of total time)
Knowledge Answer: 6.7s (26% of total time)
Embedding: 1.5s (6% of total time)
Combined Analysis: 1.3s (5% of total time)

🔍 DEBUG: Is First Message: True
🔍 DEBUG: Conversation ID from request: 
🔍 DEBUG: Generating conversation title for: should we take care of person according to constitituion?
🔍 DEBUG: Generated Title: Constitutional Care Responsibilities Discussion
🔍 DEBUG: Combined Analysis Result: {'rephrasedQuery': 'Should we take care of a person according to the constitution?', 'classification': 'KNOWLEDGE_QUESTION'}
🔍 DEBUG: Final Rephrased Query: Should we take care of a person according to the constitution?
🔍 DEBUG: Final Classification: KNOWLEDGE_QUESTION
⏱️ TIMING: Combined Analysis: 3540.96ms
🔍 DEBUG: Original Query: should we take care of person according to constitituion?
🔍 DEBUG: Rephrased Query: Should we take care of a person according to the constitution?
🔍 DEBUG: Classification: KNOWLEDGE_QUESTION
🔍 DEBUG: Conversation History Length: 0
⏱️ TIMING: Embedding Generation: 938.51ms
⏱️ TIMING: Pinecone Search: 12411.23ms
⏱️ TIMING: RAG Retrieval: 13349.97ms
🔍 DEBUG: RAG Success: True
🔍 DEBUG: Total Retrieved: 3
🔍 DEBUG: Retrieved Content Preview: --- Page 53 --- THE CONSTITUTION OF INDIA (Part IV.— Directive Principles of State Policy) 22 1[(f) 
🔍 DEBUG: Generating KNOWLEDGE answer with 3 content pieces
🔍 DEBUG: Previous Response ID: 
🔍 DEBUG: About to call OpenAI with model: gpt-5-mini
🔍 DEBUG: OpenAI Response ID: resp_0dccb603fc3534b60068ef82b8540c819fa859253249b68730
🔍 DEBUG: OpenAI Response Text Length: 1782
⏱️ TIMING: Knowledge Answer Generation: 6326.13ms
🔍 DEBUG: Knowledge Answer Generated: Yes, according to the Constitution of India, there are several provisions that emphasize the importa...
🔍 DEBUG: New Conversation ID: resp_0dccb603fc3534b60068ef82b8540c819fa859253249b68730
⏱️ TIMING: TOTAL REQUEST TIME: 24689.95ms
⏱️ TIMING BREAKDOWN:
   - Combined Analysis: 3540.96ms
   - RAG Retrieval: 13349.97ms
   - Knowledge Answer: 6326.13ms

```


## Cost Evaluation
``` 
Withou batch price
1. Assembly ai --> $0.15/hr and wishper large v3 0.111/hr
2. text-embedding-3-small --> $0.02/million
3. Pinecone: 100k user/ day to retrieve 5 docs/vector from 2 Million docs/vector will cost around $700 - $800/month same for 500k use per day will cost around $3500-4000 

Query Cost Estimation
	•	Each query fetches the top 5 results, which involves:
	•	About 5-10 Read Units (RUs) per query, depending on the namespace size.
	•	100,000 queries/day results in roughly 500,000 to 1,000,000 RUs/day.
Cost Calculation
	•	At approximately $24 per million RUs, the daily query cost is roughly:
	•	$12 - $24 per day.
	•	Monthly, this totals around $360 - $720.

4. PyMuPDF/Document AI  cost :

5.
Steps
1. Query reprhasing 
2. Query classification
3. Title generation(only for first message)
4. final answer as per general  question and knwoledge based question


1. Retrieve docs  from constitution of india are ranging from 400 to 1500(large paragraph) token(calculated by open ai tokenizer) of one 1 retrive docs. For 5 docs it can max go to 5K around. so total 5k token for retrive max but on average for one we are assuming 500 per docs to become 2500 to 3000. 
On single max 1500
5 docs ------ 5000(on Avg)

On single avg 500
5 docs ----- 3000(on avg)

so 3000 to 5000 token expect from RAG for each query.


2. 
a. General question prediction: prompt only 400 token 
b. Knowledge based question : prompt around 400 token assume but may increase and retrieve content token around 3000 to 5000 token.

3. Output: around 50 to 400 token

4. Query Reprhasing + Query Classification:  
Although it is seperate but can be combine in one to optimize the token:
Query Rephrasing prompt cost around 300 token  same for query classification 
but past converstaion may take upto 1000 token total. Need to check what need to be sent for past converation like retrive content is required or not. 

token count can be calulated for past conversation using openai input token count method. Although only past question and answer are required for past conversation(Not retrieve content is required in past conversation).











With batch price 




```







## Improvement in system along with cost optimization
``` 
1. Retrieve 30 to 50 chunk and then use cohere reranker or any re ranker to retrieve best or top 3-5
2. Use small model like gpt5-nano(or more simillar one from other) to reduce cost for small task and only use costly model for final answer.
3. 





```


## All endpoint working flow
``` 
Query
|
|
Query Reprhasing





```