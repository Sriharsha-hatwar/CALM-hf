# CALM-hf
Implementation of the paper : https://arxiv.org/abs/2401.02412

For running the code : 

```
python train.py
```

Core implementation is in the file `final_calm.py` and contains `CALMForCausalModeling` class that is inherting from PreTrainedModel and supports several functionality offered from huggingface. 

Dataset : GSM8K 

<div style="display: flex;">
    <img src="images/loss_curve.png" width="400" />
    <img src="images/perplexity_curve.png" width="400" />
</div>
