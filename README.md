# chatbot_llm

An llm powered chatbot

## Phase 1: Play with model parameters.

Using both models, try different combinations of parameters to produce the desired output for both models. Both of this models can be found in the drive folder 

- `/dream_gpt2`
- `/email_gpt2`

Those are the two folders containing the model and checkpoints of the emails and dream datasets. In both cases, we can make use of the tensors to add more data. The format required for both is an entire corpus of text in a txt format. 

- `gpt2_chat.py`


In the case of the parameters we can test, on the `bot_wcorpus.py` file there are some parameters we can move around to generate the desired output: 

Popular ones:

- **temperature** = Higher = more randomness. (higher temperature flattens the probability distribution, making rare tokens/outliers more likely. A predictable output would be something from 0.1 to 0.5. The closer to 1 is a neutral response. going higher that 1 would be more and more chaotic)
- **top_p** = Lower = limit to top percentage of probability mass. (Same as before, anything in the 0.1 to 0.6 its less random. closer to 1 is the neutral one, open to rare words. 1 just means no filtering, so it relies on temperature.
- **top_k** = Limit to top K most likely tokens.
- **repetition_penalty** = Higher = less repeating.
- **max_length** = How long the output can be.
- **no_repeat_ngram_size** = Dont repeat phrases of this length.
- **do_sample** = If True, it samples tokens instead of taking argmax.

Important links with detaailed explanations:
- [Tutorial on the difference of each parameter](https://machinelearningmastery.com/understanding-text-generation-parameters-in-transformers/)
- [Found this in a git issue, more probability](https://huggingface.co/blog/how-to-generate)
- [Really good explanation of the intersection of temp and top_k p](https://medium.com/@1511425435311/understanding-openais-temperature-and-top-p-parameters-in-language-models-d2066504684f)

All parameters (there are tons):

[Official hugging face function description](https://huggingface.co/docs/transformers/main_classes/text_generation)

