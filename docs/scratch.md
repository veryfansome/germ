# Notes
- design a game and train a model to play the game
- transformer trained via RLHF 
- novel user behavior should be interesting
- copying observed novel user behavior should be "instinctual"
- rewarded on positive user feedback 
- maybe there needs to be some reward hierarchy, i.e. physiological, safety, social, and esteem needs so that the agent 
  is incentivized to pursue more complex behaviors
- use scripted conversations to learn when not with user
- an initial experiment should be to see if this way of learning text generation using a vector-controlled switch is even possible
- experiment 1: given a semantic target, generate text that gets as similar to the target as possible

## Game

## Model outputs / Possible agent actions
```text
------|
Slot  | 
------|
s0    | <- boolean for buffer or send, which represents end-of-thought
------|
s1    | <- intention embedding of next word for vector-controlled switch
------|
```

## Model inputs / States

### Text input / output buffer
- an internal state
- per-word sliding window
- shapes next text chunk out
- shows the impact of each additional word or token
```text
-------------------------|----|----|----|----|----|----|
Input type               | s0 | s1 | s2 | s3 | s4 | sN |
-------------------------|----|----|----|----|----|----|
word embedding           |    |    |    |    |    |    | <- embedding of last word in buffer
-------------------------|----|----|----|----|----|----|
word embedding (norm)    |    |    |    |    |    |    | <- normalized embedding of last word in buffer
-------------------------|----|----|----|----|----|----|
summary embedding        |    |    |    |    |    |    | <- summary embedding of all words in buffer
-------------------------|----|----|----|----|----|----|
summary embedding (norm) |    |    |    |    |    |    | <- normalized summary embedding of all words in buffer. Each
-------------------------|----|----|----|----|----|----|    incremental word should have a diminishing affect on the
                                                            overall meaning
```

### Chat history buffer
- external state of conversation
- per-message sliding window
- stores semantic directions in conversation history
- shows the impact of each additional message
```text
-------------------------|----|----|----|----|----|----|
Input type               | s0 | s1 | s2 | s3 | s4 | sN |
-------------------------|----|----|----|----|----|----|
message embedding        |    |    |    |    |    |    | <- summary embedding of last message in the conversation
-------------------------|----|----|----|----|----|----|
message embedding (norm) |    |    |    |    |    |    | <- normalized summary embedding of last message in the conversation
-------------------------|----|----|----|----|----|----|
agent or user            |    |    |    |    |    |    | <- boolean indicating ownership of last message in chat window
-------------------------|----|----|----|----|----|----|
```

### vector-controlled switch:
- vector index with one or more text-based anchors that correspond to actions the agent can learn and perform
- given a set of embedding inputs, the agent has to learn how to generate an output embedding that triggers learned action
- once the generated vector is similar enough to an anchor, the action associated with that anchor is triggered
- triggered actions can alter the external environment which alters input embeddings
- normalized vectors exist on a sphere and can hold an infinite number of possible "directions", each corresponding with some intent
- input embeddings can maybe model human working memory
- input embeddings describe the agent's the external environment and serve to convey the external state
- each trigger corresponds with a model that learns it's anchor's thresholds

```python
import asyncio
import faiss
import logging
import numpy as np
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

async_openai_client = AsyncOpenAI()


class VectorControlledSwitch:
    def __init__(
            self,
            embedding_dimensions: int = 3072,  # text-embedding-3-large can be shortened to 256
    ):
        self.embedding_dimensions = embedding_dimensions
        self.vector_switch = faiss.IndexIDMap(faiss.IndexFlatIP(embedding_dimensions))
        self.actions = []
    
    async def add_action_vector(self, desc: str, action):
        self.actions.append(action)
        action_id = self.actions.index(action)
        await asyncio.to_thread(
            self.vector_switch.add_with_ids,
            await self.get_text_vector(desc), np.array([action_id], dtype=np.int64)
        )

    async def get_text_vector(self, text: str):
        response = await async_openai_client.embeddings.create(
            dimensions=self.embedding_dimensions,
            model="text-embedding-3-large",
            input=text,
            encoding_format="float"
        )
        vector = np.array([response.data[0].embedding], dtype=np.float32)
        faiss.normalize_L2(vector)  # Important for cosine search
        return vector

    async def tick(self, intention_vector, min_sim_score: float = 0.999):
        sim_scores, neighbors = await asyncio.to_thread(self.vector_switch.search, intention_vector, 1)
        for rank, (action_id, sim_score) in enumerate(zip(neighbors[0], sim_scores[0]), 1):
            if action_id != -1 and sim_score > min_sim_score:  # -1 means no match
                logger.info(f"{rank:>2}. action_id={action_id} sim={sim_score:.4f}")
                self.actions[action_id]()
```