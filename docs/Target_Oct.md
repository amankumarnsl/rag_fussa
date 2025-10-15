# 3 Target

### Cost Evaluation

### All endpoint working flow

### Improvement in system along with cost optimization

# ------------------------------


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