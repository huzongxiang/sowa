from typing import Tuple, List, Dict, Union
from functools import partial
import torch
import torch.nn as nn
from src.models.components.clip.simple_tokenizer import SimpleTokenizer


def tokenize(texts: Union[str, List[str]], tokenizer: SimpleTokenizer, context_length: int = 77, truncate: bool = False) -> torch.Tensor:
    """
    Tokenizes input texts using a predefined tokenizer.

    Args:
        texts: A string or a list of strings to be tokenized.
        context_length: The fixed context length for tokenization. Default is 77, as used in CLIP.
        truncate: Whether to truncate the texts if they exceed the context length.

    Returns:
        A tensor of tokens with shape (number of texts, context_length).
    """
    # Implementation details remain unchanged
    if isinstance(texts, str):
        texts = [texts]

    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)
    
    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class PromptTextEncoder(nn.Module):
    def __init__(self, clip_model: nn.Module, prompt_learner: nn.Module, text_encoder: Union[nn.Module, None]=None):
        """
        Initializes the TextFeatureExtractor.

        Args:
            prompt_learner (nn.Module): The prompt learner module.
            text_encoder (nn.Module, optional): The text encoder module, required if using a text encoder other than AnomalyPromptLearner.
        """
        super().__init__()
        self.prompt_learner = prompt_learner(clip_model)
        self.class_names = self.prompt_learner.class_names if hasattr(self.prompt_learner, "class_names") else None

        self.text_encoder = text_encoder
        if isinstance(text_encoder, partial):
            self.text_encoder = text_encoder(clip_model)

        # Create a mapping from class names to indices
        if self.class_names is not None:
            self.class_name_to_idx = {name: idx for idx, name in enumerate(self.class_names)}

    def forward(self, cls_names: List[str]) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Generates text features from prompts.

        Returns:
            Tuple containing tensor of text features and list of tensors for additional text feature maps.
        """
        if isinstance(self.text_encoder, nn.Module):
            prompts, tokenized_prompts = self.prompt_learner()
            text_features, text_features_maps = self.text_encoder(prompts, tokenized_prompts)
        else:
            text_features = self.prompt_learner()
            text_features_maps = []

        pooled_text_features = self.pool_text_features(text_features)  # Pool to (M, 2, C)
        text_features = self.get_batch_text_features(pooled_text_features, cls_names)  # (B, 2, C)

        return text_features, text_features_maps

    def pool_text_features(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Pool text features within normal and anomaly categories to get shape (M, 2, C).

        Args:
            text_features: Tensor of shape (2 * M * K, C).

        Returns:
            Pooled text features of shape (M, 2, C).
        """
        M = len(self.class_names)
        K = text_features.size(0) // (2 * M)

        pooled_text_features = text_features.view(2, M, K, -1).mean(dim=2).permute(1, 0, 2)  # (M, 2, C)
        return pooled_text_features

    def get_batch_text_features(self, pooled_text_features: torch.Tensor, cls_names: List[str]) -> torch.Tensor:
        """
        Get text features for a batch of class names.

        Args:
            pooled_text_features: Tensor of shape (M, 2, C).
            cls_names: List of class names for corresponding features.

        Returns:
            Batch text features of shape (B, 2, C).
        """
        cls_indices = [self.class_name_to_idx.get(cls_name, 0) for cls_name in cls_names]
        batch_text_features = pooled_text_features[cls_indices]    # (B, 2, C)
        return batch_text_features


class PromptEncoder(nn.Module):
    """
    Encodes text prompts for anomaly detection using a pretrained model.
    
    Args:
        model (nn.Module): The pretrained model for encoding text.
        tokenizer (SimpleTokenizer): The tokenizer used for tokenizing text.
        class_names (List[str]): List of class names to be used in the prompts.
        prompt_normal (List[str]): List of normal state prompts.
        prompt_abnormal (List[str]): List of abnormal state prompts.
        prompt_templates (List[str]): List of templates to generate the full prompts.
        context_length (int): The context length for tokenization. Default is 77.
        truncate (bool): Whether to truncate the texts if they exceed the context length. Default is False.
    """
    
    def __init__(
        self, 
        model: nn.Module, 
        tokenizer: SimpleTokenizer,
        class_names: List[str],
        prompt_normal: List[str], 
        prompt_abnormal: List[str], 
        prompt_templates: List[str],
        context_length: int = 77,
        truncate: bool = False
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.class_names = class_names
        self.prompt_normal = prompt_normal
        self.prompt_abnormal = prompt_abnormal
        self.prompt_templates = prompt_templates

        # Initialize the tokenize function with partial application
        self.tokenize = partial(
            tokenize,
            tokenizer=tokenizer,
            context_length=context_length,
            truncate=truncate,
        )

        self.encode_prompt()

    def encode_prompt(self):
        """
        Encodes the text prompts and stores the resulting text features in a buffer.
        """
        use_cuda = torch.cuda.is_available()
        device = torch.device("cuda:0" if use_cuda else "cpu")
        self.model.to(device)

        with torch.no_grad():
            prompt_state = [self.prompt_normal, self.prompt_abnormal]
            text_features = []
            for state_prompts in prompt_state:
                state_features = []
                for obj in self.class_names:
                    prompted_state = [state.format(obj) for state in state_prompts]
                    prompted_sentence = []
                    for s in prompted_state:
                        for template in self.prompt_templates:
                            prompted_sentence.append(template.format(s))
                    prompted_sentence = self.tokenize(prompted_sentence).to(device)
                    class_embeddings = self.model.encode_text(prompted_sentence)
                    class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                    class_embedding = class_embeddings.mean(dim=0)
                    class_embedding /= class_embedding.norm()
                    state_features.append(class_embedding)

                # Stack the features for each state (normal/abnormal)
                state_features = torch.stack(state_features)
                text_features.append(state_features)                               # (2, M, C)

            # Stack the text features for both states and reshape
            text_features = torch.stack(text_features).view(len(prompt_state) * len(self.class_names), -1)     # (2 * M, C)

        # Register the text features as a buffer
        self.register_buffer('text_features', text_features)

    def forward(self):
        """
        Returns the precomputed text features.
        
        Returns:
            torch.Tensor: The precomputed text features.
        """
        return self.text_features
    

class AnomalyPromptLearner(nn.Module):
    """
    Learns prompts for anomaly detection using a pretrained CLIP model.
    This class handles the creation, tokenization, and embedding of textual prompts.
    """
    def __init__(
        self, 
        clip_model: nn.Module,
        tokenizer: SimpleTokenizer,
        class_names: List[str]=['object'],
        prompt_length: int=12,
        context_length: int=77,
        truncate: bool=False,
        state_template: Dict={"normal": ["{}"], "anomaly": ["damaged {}"]},
    ):
        super().__init__()
        self.clip_model = clip_model
        self.class_names = class_names
        self.prompt_length = prompt_length

        embedding_dim = clip_model.ln_final.weight.shape[0] 
        self.embedding_dim = embedding_dim

        self.tokenize = partial(
            tokenize,
            tokenizer=tokenizer,
            context_length=context_length, 
            truncate=truncate,
        )

        # State lists
        self.state_normal_list: List[str] = state_template["normal"]
        self.state_anormaly_list: List[str] = state_template["anomaly"]

        # Initialize prompts
        self.prompt_prefix_pos = " ".join(["X"] * prompt_length)
        self.prompt_prefix_neg = " ".join(["X"] * prompt_length)

        # Initialize parameters for prompt embeddings
        self.ctx_pos = nn.Parameter(
            torch.empty(
                len(self.class_names),
                len(self.state_normal_list),
                prompt_length,
                embedding_dim)
        )
        self.ctx_neg = nn.Parameter(
            torch.empty(
                len(self.class_names),
                len(self.state_anormaly_list),
                prompt_length,
                embedding_dim)
        )

        self.reset_parameters()
        self.generate_and_register_prompts()

    def reset_parameters(self):
        """
        Initiate weights of learnable prompts.
        """
        nn.init.normal_(self.ctx_pos, std=0.02)
        nn.init.normal_(self.ctx_neg, std=0.02)

    def tokenize_and_embed(self, prompts: List[str]) -> torch.Tensor:
        """
        Tokenizes and embeds given list of text prompts.
        """
        tokenized = [self.tokenize(prompt) for prompt in prompts]
        tokenized = torch.cat(tokenized)                         # Shape: [len(prompts), self.prompt_length]
        embeddings = self.clip_model.token_embedding(tokenized)  # Shape: [len(prompts), embedding_dim]
        return tokenized, embeddings

    def generate_and_register_prompts(self) -> torch.Tensor:
        """
        Generates embedded prompts for both normal and damaged cases.
        """
        prompts_pos = []
        prompts_neg = []

        # Inportance to reshape or view keep dimension consistent
        # Outer-loop for classes
        for name in self.class_names:
            # Inner-loop for states
            for template in self.state_normal_list:
                prompt = f"{self.prompt_prefix_pos} {template.format(name)}."
                prompts_pos.append(prompt)
            
            for template in self.state_anormaly_list:
                prompt = f"{self.prompt_prefix_neg} {template.format(name)}."
                prompts_neg.append(prompt)

        # Embed prompts
        tokenized_pos, embeddings_pos = self.tokenize_and_embed(prompts_pos)
        tokenized_neg, embeddings_neg = self.tokenize_and_embed(prompts_neg)

        # Reshape tokens to [num_classes, num_states, prompt_length]
        tokenized_pos = tokenized_pos.view(len(self.class_names), len(self.state_normal_list), -1)
        tokenized_neg = tokenized_neg.view(len(self.class_names), len(self.state_anormaly_list), -1)

        # Reshape embeddings to [num_classes, num_states, prompt_length, embedding_dim]
        _, L, D = embeddings_pos.shape
        embeddings_pos = embeddings_pos.view(len(self.class_names), len(self.state_normal_list), L, D)
        embeddings_neg = embeddings_neg.view(len(self.class_names), len(self.state_anormaly_list), L, D)

        token_prefix_pos = embeddings_pos[:, :, :1, :] 
        token_suffix_pos = embeddings_pos[:, :, 1 + self.prompt_length:, :]
        token_prefix_neg = embeddings_neg[:, :, :1, :]
        token_suffix_neg = embeddings_neg[:, :, 1 + self.prompt_length:, :]

        self.register_buffer('tokenized_prompts_pos', tokenized_pos)
        self.register_buffer('tokenized_prompts_neg', tokenized_neg)
        self.register_buffer('token_prefix_pos', token_prefix_pos)
        self.register_buffer('token_suffix_pos', token_suffix_pos)
        self.register_buffer('token_prefix_neg', token_prefix_neg)
        self.register_buffer('token_suffix_neg', token_suffix_neg)
        
    def forward(self, class_indices: torch.Tensor=None) -> torch.Tensor:
        """
        Forward pass to select the correct prompts based on class indices and damage condition.
        """
        # Gather the correct embeddings based on the class and damage status
        prompts_pos = torch.cat([self.token_prefix_pos, self.ctx_pos, self.token_suffix_pos], dim=2)
        prompts_neg = torch.cat([self.token_prefix_neg, self.ctx_neg, self.token_suffix_neg], dim=2)

        # Reshape for easy selection
        _, _, L, D = prompts_pos.shape
        prompts_pos = prompts_pos.view(-1, L, D)                                # (M * K, 77, C)      context_length: 77
        prompts_neg = prompts_neg.view(-1, L, D)                                # (M * K, 77, C)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)                  # (2 * M * K, 77, C)

        # Reshape tokens
        _, _, L = self.tokenized_prompts_pos.shape
        tokenized_pos = self.tokenized_prompts_pos.view(-1, L)                  # (M * K, 77)         context_length: 77
        tokenized_neg = self.tokenized_prompts_neg.view(-1, L)                  # (M * K, 77)
        tokenized_prompts = torch.cat([tokenized_pos, tokenized_neg], dim=0)    # (2 * M * K, 77)

        # Compute the mean across the prompt length for simplification (optional)
        return prompts, tokenized_prompts
