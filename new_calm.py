import torch
import copy
import math
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Union, Tuple, List
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GemmaConfig,
    AutoConfig,
    BitsAndBytesConfig,
    GPT2Tokenizer)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from torch.nn import CrossEntropyLoss 
from transformers.utils import logging
from datasets import load_dataset


from transformers import (GPT2LMHeadModel, 
                        Trainer, 
                        TrainingArguments
                        )


logger = logging.get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to(device)
    print("Size of attn_bias : ", attn_bias.shape)
    print("Size of attn_mask : ", attn_mask.shape)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class CALMPreTrainedModel(PreTrainedModel):
    # ISSUE 1 - config_class - what config to write here? 
    config_class = GemmaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keep_in_fp32_modules = ["inv_freq", "rotary_emb", "cos_cached", "sin_cached"]
    _no_split_modules = ["GemmaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "causal_mask"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TransformerCompositionBlock(nn.Module):
    def __init__(self, anchor_model_config, augment_model_config):
        # This is needed to project between the augment and anchor LM according to paper.
        super().__init__()
        if anchor_model_config.hidden_size != augment_model_config.hidden_size:
            self.projection_block = nn.Linear(augment_model_config.hidden_size, anchor_model_config.hidden_size)
        else:
            self.projection_block = None
        
        self.num_attention_heads = anchor_model_config.num_attention_heads
        # You need to have num_attention_heads number of multi-headed attention..

        self.single_head_dim = anchor_model_config.hidden_size//self.num_attention_heads
        self.model_dimension = anchor_model_config.hidden_size
        
        # These are W_q, W_k, W_v matrices.
        self.Wq = nn.Linear(anchor_model_config.hidden_size, anchor_model_config.hidden_size)
        self.Wk = nn.Linear(anchor_model_config.hidden_size, anchor_model_config.hidden_size)
        self.Wv = nn.Linear(anchor_model_config.hidden_size, anchor_model_config.hidden_size)
        
        self.Wo = nn.Linear(anchor_model_config.hidden_size, anchor_model_config.hidden_size)

        # Below code is not needed, but still doing xavier initialize, can be removed.
        #torch.nn.init.xavier_normal_(self.Wq)
        #torch.nn.init.xavier_normal_(self.Wk)
        #torch.nn.init.xavier_normal_(self.Wv)
        #torch.nn.init.xavier_normal_(self.Wo)
    
    def split_heads(self, vecs, batch_size):
        vecs = vecs.view(batch_size, -1, self.num_attention_heads, self.single_head_dim)
        return vecs.transpose(1, 2)

    def forward(self, input_vecs_b, input_vecs_a, mask):
        # NOTE :  
        # input_vecs_b is from the anchor_model 
        # input_vecs_a is from augment_model
        # Shape of tensors : (batch_size, seq_len, dim)
        # You need to use a Projection matrix to project the augment model to be of same size.

        batch_size = input_vecs_a.size(0)

        # This will project the matrix to the approproate dimension..
        if self.projection_block is not None:
            input_vecs_a = self.projection_block(input_vecs_a)

        # Do linear projection and make sure : 
        # Query is created from the output of the anchor model
        # Key and Value is created from the output of the augmented model
        query = self.Wq(input_vecs_b)
        key = self.Wk(input_vecs_a)
        value = self.Wv(input_vecs_a)

        print("Query size : ", query.shape)
        print("key shape : ", key.shape)
        print("value shape : ", value.shape)

        # Dimension - (bs, seq_len, emb_dim)
        # after split head - (bs, seq_len, no_head, emb_dim)?

        # Split it into multiple heads.
        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)
        

        # Not adding any rotatory embedding over here and not mentioned in the paper as well.

        # The mask should mask out the next tokens and should let the previour tokens.
        #scores, attention_weights = ScaledDotProductAttention()(query, key, value, mask)
    
        # Conver the mask to bool..
        mask = mask.bool().to(device)
        print("Query size : ", query.shape)
        print("key shape : ", key.shape)
        print("value shape : ", value.shape)

        with torch.backends.cuda.sdp_kernel(enable_math=False):
            # scores = F.scaled_dot_product_attention(
            #     query,
            #     key,
            #     value,
            #     attn_mask=mask 
            #     )
            scores = scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=mask
            )
        
        # Concatenate heads
        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.model_dimension)

        # Final linear layer, W_o
        output = self.Wo(scores)

        return output

class CALMModel(CALMPreTrainedModel):
    def __init__(self, anchor_model, augementing_model, num_cross_over, do_quantize, config):
        # Make sure the config is from the anchor model.
        super().__init__(config)
        self.do_quantize = do_quantize
        self.anchor_model = self._model_init(anchor_model, do_quantize)
        self.augmenting_model = self._model_init(augementing_model, do_quantize)

        self.embed_tokens = self.anchor_model.get_input_embeddings()

        self.num_cross_over = num_cross_over
        # Taken this from the original implementation of https://github.com/lucidrains/CALM-pytorch
        if self.anchor_model.config.num_hidden_layers % self.num_cross_over != 0  or self.augmenting_model.config.num_hidden_layers % self.num_cross_over != 0 :
            print("Number of hidden layers in anchor model : ", self.anchor_model.config.num_hidden_layers)
            print("Number of hidden layers in augment model : ", self.augmenting_model.config.num_hidden_layers)
            raise ValueError(
                "Please specify the cross over layer to be a perfect divisible by the total number of layers in both the models."
            )
        self.multi_head_composition_blocks = nn.ModuleList([TransformerCompositionBlock(self.anchor_model.config, self.augmenting_model.config) for i in range(num_cross_over)])
        # This is false..
        self.gradient_checkpointing = False
        # Trying to get the starting point from where the blocks get the representation from anchor first
        self.anchor_layer_start_index = (self.anchor_model.config.num_hidden_layers // self.num_cross_over) 
        self.augmenting_model_start_index = (self.augmenting_model.config.num_hidden_layers // self.num_cross_over) 

        # Get all the indices where the blocks are interwind between anchor and augment
        self.anchor_model_layer_indices = [l for l in range(self.anchor_layer_start_index-1, self.anchor_model.config.num_hidden_layers, self.anchor_layer_start_index)]
        self.augmenting_model_layer_indices = [l for l in range(self.augmenting_model_start_index-1, self.augmenting_model.config.num_hidden_layers, self.augmenting_model_start_index)]
        print("Printing out the layer indices from anchor model : ", self.anchor_model_layer_indices)
        print("Printing out the layer indices from augment model", self.augmenting_model_layer_indices)

        # Depending on the type of model used, the norm is used.. so trying to see if I can directly
        # access the norm from anchor model - BAD CODE AHEAD, accessing private variables directly!!

        # Or should I need to train this layer norm?!! - CONFUSING 1 
        self.norm = self.anchor_model.model.norm

        # Write code for freezing all the layers in both anchor and augmenting model.
        self.freeze_all_layers_(self.anchor_model)
        self.freeze_all_layers_(self.augmenting_model)
    
    def freeze_all_layers_(self, module):
        self.set_module_requires_grad_(module, False)
    
    def set_module_requires_grad_(
        self,
        module: Module,
        requires_grad: bool
    ):
        
        for param in module.parameters():
            param.requires_grad = requires_grad
    
    def _model_init(self, model_name, do_quantize):
        bnb_config= BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16)

        if do_quantize:
            model = AutoModelForCausalLM.from_pretrained(model_name, 
                            quantization_config=bnb_config,
                            attn_implementation = "flash_attention_2",
                            cache_dir = "/work/pi_dhruveshpate_umass_edu/shatwar/.cache/hub")
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name,
                    attn_implementation = "flash_attention_2",
                    cache_dir = "/work/pi_dhruveshpate_umass_edu/shatwar/.cache/hub",
                    torch_dtype=torch.float16).to("cuda")
        return model

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def combine_with_other_states(self, layer_output, history, block_count, mask) :
        # Combine the layer_output with whatever we get from forward done with 
        # layer_output in paper is H_bj and .. 
        # history has - all_hidden_states, so choose from
        #print("Block count is : ", block_count)
        index = self.augmenting_model_layer_indices[block_count]
        #print("The output that needs to be taken from index :", index)
        augment_model_output = history.hidden_states[self.augmenting_model_layer_indices[block_count]]
        with torch.cuda.amp.autocast():
            output = self.multi_head_composition_blocks[block_count](layer_output, augment_model_output, mask)
        return output
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        # Might need to change somethings below..
        history = self.augmenting_model(
           input_ids,
           attention_mask,
           position_ids,
           past_key_values,
           inputs_embeds,
           use_cache,
           output_attentions,
           output_hidden_states=True,
           return_dict=return_dict,
           cache_position=cache_position
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):  # kept for BC (non `Cache` `past_key_values` inputs)
            return_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # Very bad Code -----
        causal_mask = self.anchor_model.model._update_causal_mask(
            attention_mask, 
            inputs_embeds, 
            cache_position, 
            past_key_values, 
            output_attentions
        )

        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # Gemma downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        composition_block_count = 0

        for index, decoder_layer in enumerate(self.anchor_model.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # removing the part where we can do training and gradient checkpointing.
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            if index in self.anchor_model_layer_indices:
                # This gets the inputs from the multi_head_composition_blocks.
                # You need to figure out about MASK..
                hidden_states = self.combine_with_other_states(
                    layer_outputs[0], 
                    history,
                    composition_block_count,
                    attention_mask
                    )
                composition_block_count += 1
                # This is the residual from the block and the previous decoder layer.
                hidden_states = hidden_states + layer_outputs[0]
            else:
                hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class CALMForCausalModeling(CALMPreTrainedModel):
    # Doing this as its there in several Decoder only models. Gemma, LLama, GPT-2
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, anchor_model, augementing_model, num_cross_over, do_quantize, config):
        super().__init__(config)
        self.model = CALMModel(anchor_model, augementing_model, num_cross_over, do_quantize, config)
        self.vocab_size = config.vocab_size
        # The below lm_head should also be from anchor model.. so why not directly access
        # that from anchor model!! - BAD CODE AHEAD... 
        self.lm_head = self.model.anchor_model.lm_head
    
        # I dont need to do any initializations, but to keep it in sync with other Decoder
        # models, I am doing this.
        # self.init_weights()
    
    def get_input_embeddings(self):
        # Need to change this to get it from anchor model.
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        # Need to change this to set it to anchor model.
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # Need to change this to set it to anchor model?  
        self.model = decoder
    
    def get_anchor_model(self):
        return self.model.anchor_model

    def get_augment_model(self):
        return self.model.augmenting_model

    def get_decoder(self):
        # Need to change this to return anchor model??
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism - 
            # ISSUE 2 - is this needed?
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
    
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    # This method is directly copied from modeling_llama, not sure where it will be used
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
    
class MathDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        # Labels are usually the input_ids themselves for language modeling
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = item['input_ids'].clone()
        return item

    
def only_gpt():
    # This code is to train GPT-2 and see how to optimize trainer arguments..
    print("Running only GPT-2 code..")
    train_csv = pd.read_csv("gsm8k/main_train.csv")
    test_csv = pd.read_csv("gsm8k/main_test.csv")
    training_examples = ["Here is a Math question : "+ q + " Answer: " + a for q, a in zip(train_csv["question"], train_csv["answer"])]
    test_examples = ["Here is a Math question : "+ q + " Answer: " + a for q, a in zip(test_csv["question"], test_csv["answer"])]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    train_encodings = tokenizer(training_examples, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_examples, truncation=True, padding=True, max_length=512)
    train_feats = {}
    val_feats = {}
    last_index = int(len(train_encodings["input_ids"]) * 0.9)
    train_feats["input_ids"] = train_encodings["input_ids"][:last_index]
    train_feats["attention_mask"] = train_encodings["attention_mask"][:last_index]
    val_feats["input_ids"] = train_encodings["input"][last_index:]
    val_feats["attention_mask"] = train_encodings["attention_mask"][last_index:]

    model = GPT2LMHeadModel.from_pretrained('gpt2')




    # training_args = TrainingArguments(
    #     output_dir='./results',          # Output directory
    #     evaluation_strategy="epoch",     # or use "steps" to evaluate every "eval_steps"
    #     eval_steps=steps_for_50_percent,                  # if using steps, set how often to evaluate
    #     num_train_epochs=3,              # Number of training epochs
    #     per_device_train_batch_size=4,   # Batch size for training
    #     per_device_eval_batch_size=4,    # Batch size for evaluation
    #     logging_dir='./logs',            # Directory for storing logs
    #     logging_steps=steps_for_50_percent,               # How often to log loss
    #     save_strategy="epoch",           # Save strategy
    #     load_best_model_at_end=True,     # Load the best model at the end of training
    # )

    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    # )

    # trainer.train()
    # dataset = load_dataset("juletxara/mgsm", 'te')
    # print(dataset)
    # print(dataset["test"][4])
    
    # augment_model_name = "google/gemma-2b-it"
    # anchor_model_name = "google/gemma-2b-it"

    # bnb_config= BitsAndBytesConfig(
    #             load_in_4bit=True,
    #             bnb_4bit_quant_type='nf4',
    #             bnb_4bit_use_double_quant=True,
    #             bnb_4bit_compute_dtype=torch.bfloat16)

    # augment_model = AutoModelForCausalLM.from_pretrained(augment_model_name, 
    #                         quantization_config=bnb_config,
    #                         attn_implementation = "flash_attention_2",
    #                         cache_dir = "/work/pi_dhruveshpate_umass_edu/shatwar/.cache/hub")
    # anchor_model = AutoModelForCausalLM.from_pretrained(anchor_model_name, 
    #                         quantization_config=bnb_config,
    #                         attn_implementation = "flash_attention_2",
    #                         cache_dir = "/work/pi_dhruveshpate_umass_edu/shatwar/.cache/hub")

    # some = TransformerCompositionBlock(anchor_model.config, augment_model.config).to(device)
    
    # input_vecs_b = torch.rand(2, 512, 2048).to(device)
    # input_vecs_a = torch.rand(2, 512, 2048).to(device)
    # ones = torch.ones(230)
    # zeros = torch.zeros(282)
    # mask = torch.cat((ones, zeros))
    # mask = mask.bool().to(device)
    # with torch.cuda.amp.autocast():
    #     some(input_vecs_b, input_vecs_a, mask)

if __name__ == "__main__":
    # augment_model = "google/gemma-2b-it"
    # anchor_model = "google/gemma-2b-it"
    # num_cross_over = 3
    # quantize = True
    # #calm_config = BitsAndBytesConfig(load_in_8bit=False)
    # tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b")
    # calm_config = AutoConfig.from_pretrained("google/gemma-2b-it")
    # llm_aug_llm = CALMForCausalModeling(
    #     anchor_model,
    #     augment_model,
    #     num_cross_over,
    #     quantize,
    #     calm_config
    # ).to(device)
    # text = "Hello, this is an example text to tokenize. The purpose of this text is to demonstrate how to prepare data for a text generation model."
    # max_length = llm_aug_llm.get_anchor_model().config.max_position_embeddings
    # #print("max length : ", max_length)
    # encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=max_length)
    # input_ids = encoded_input['input_ids'].to(device)
    # labels = input_ids.clone().to(device)
    # attention_mask = encoded_input['attention_mask'].to(device)
    # output = llm_aug_llm(
    #     input_ids = input_ids,
    #     labels = labels,
    #     attention_mask = attention_mask
    # )
    # print("output : ", output.loss)
    only_gpt()
