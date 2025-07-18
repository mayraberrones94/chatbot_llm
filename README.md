# chatbot_llm

An llm powered chatbot

## Phase 1: Play with model parameters.

Using both models, try different combinations of parameters to produce the desired output for both models.

- `/dream_gpt2`
- `/email_gpt2`

Those are the two folders containing the model and checkpoints of the emails and dream datasets. In both cases, we can make use of the tensors to add more data. The format required for both is an entire corpus of text in a txt format. 

- `bot_wcorpus.py`

This is the python file that we can use in different ways:

- `python bot_wcorpus.py`: It would prompt the user to write manually the interactions with the model.
- `python bot_wcorpus.py --model_path ./path_to_folder`: It would open a specific model and the user would have to write the prompts manually. Default mode for the time being is dream_gpt2, since its the first model to be ready to test.
- `python bot_wcorpus.py --prompt_file name_of_file.txt`: it would then go through each line of the txt file, so each line is a different prompt.
- `python bot_wcorpus.py --output_file name_of_file.txt`: Optional, and it would save all of the interactons with the bot once its done.

We can also do a combination of all if needed.

In the case of the parameters we can test, on the `bot_wcorpus.py` file there are some parameters we can move around to generate the desired output: (From line 8 to 28)

Popular ones:

- **temperature** = Higher = more randomness.
- **top_p** = Lower = limit to top % of probability mass.
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

