import unittest

from transformers import is_torch_available, BlenderbotTokenizer, BlenderbotConfig
from transformers.file_utils import cached_property

# from .utils import require_torch, slow, torch_device
from transformers.testing_utils import require_torch, slow, torch_device
from transformers.tokenization_blenderbot import BlenderbotSmallTokenizer
from .test_configuration_common import ConfigTester
from .test_modeling_common import ModelTesterMixin, ids_tensor


# parlai import to test Blenderbot outputs agains parlai (will be removed at the end)

if is_torch_available():
    import torch
    from transformers import (
        BlenderbotForConditionalGeneration,
    )



@require_torch
class BlenderbotModelTester:
    def __init__(
        self,
        parent,
        batch_size=2,
        seq_len=10,
        vocab_size=100,
        hidden_size=16,
        is_training=False,
        num_hidden_layers=2,
        num_attention_heads=2,
        hidden_dropout_prob=0.2,
        max_position_embeddings=50,
        eos_token_id=2,
        bos_token_id=0,
        pad_token_id=1,
        use_labels=True,
        ffn_size=4,
        attention_dropout=0.2,
        activation="gelu",
    ):
        self.parent = parent
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.is_training = is_training
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings
        self.use_labels = use_labels
        self.ffn_size = ffn_size
        self.activation = activation
        self.attention_dropout = attention_dropout
        torch.manual_seed(0)

    def prepare_config_and_inputs_for_common(self):
        input_ids = ids_tensor([self.batch_size, self.seq_len], self.vocab_size)

        config = BlenderbotConfig(
            d_model=self.hidden_size,
            dropout=self.hidden_dropout_prob,
            vocab_size=self.vocab_size,
            encoder_layers=self.num_hidden_layers,
            decoder_layers=self.num_hidden_layers,
            encoder_attention_heads=self.num_attention_heads,
            decoder_attention_heads=self.num_attention_heads,
            attention_dropout=self.attention_dropout,
            encoder_ffn_dim=self.ffn_size,
            decoder_ffn_dim=self.ffn_size,
            max_position_embeddings=self.max_position_embeddings,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
            num_beams=1,
            min_length=3,
            max_length=10,
        )
        attention_mask = ids_tensor([self.batch_size, self.seq_len], vocab_size=2)
        inputs_dict = {"input_ids": input_ids, "attention_mask": attention_mask}
        return config, inputs_dict


@require_torch
class BlenderbotTesterMixin(ModelTesterMixin, unittest.TestCase):
    all_generative_model_classes = (BlenderbotForConditionalGeneration,) if is_torch_available else ()

    is_encoder_decoder = True
    test_head_masking = False
    test_pruning = False
    test_torchscript = False
    test_resize_embeddings = True
    test_missing_keys = False

    def setUp(self):
        self.model_tester = BlenderbotModelTester(self)
        self.config_tester = ConfigTester(self, config_class=BlenderbotConfig)

    def test_inputs_embeds(self):
        pass

    def test_initialization_module(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotForConditionalGeneration(config)
        model.to(torch_device)
        model.eval()
        self.assertTrue((model.encoder.embed_tokens.weight == model.shared.weight).all().item())
        self.assertAlmostEqual(torch.std(model.encoder.embed_tokens.weight).item(), config.init_std, 2)
        self.assertAlmostEqual(torch.std(model.encoder.embed_positions.weight).item(), config.init_std, 2)

    def test_embed_pos_shape(self):
        config, inputs_dict = self.model_tester.prepare_config_and_inputs_for_common()
        model = BlenderbotForConditionalGeneration(config)
        expected_shape = (config.max_position_embeddings, config.d_model)
        self.assertEqual(model.encoder.embed_positions.weight.shape, expected_shape)
        self.assertEqual(model.decoder.embed_positions.weight.shape, expected_shape)

from parlai.agents.transformer.modules import MultiHeadAttention, TransformerDecoderLayer, TransformerEncoder, TransformerGeneratorModel
@require_torch
class AbstractBlenderBotIntegrationTests(unittest.TestCase):
    checkpoint_name = "sshleifer/blenderbot-3B"
    tokenizer_cls = BlenderbotTokenizer
    @cached_property
    def model(self):
        model = BlenderbotForConditionalGeneration.from_pretrained(self.checkpoint_name).to(torch_device)
        if torch_device == "cuda":
            model = model.half()
        return model

    @cached_property
    def tokenizer(self):
        return self.tokenizer_cls.from_pretrained(self.checkpoint_name)


class Blenderbot3BIntegrationTests(AbstractBlenderBotIntegrationTests):
    @slow
    def test_generation_same_as_parlai_3B(self):
        src_text = [
            "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like i'm going to throw up.\nand why is that?"
        ]
        tgt_text = ["I'm not sure, but I do know that social anxiety disorder is a mental disorder"]
        model_inputs = self.tokenizer(src_text, return_tensors='pt').to(torch_device)
        generated_utterances = self.model.generate(**model_inputs)
        self.assertListEqual(tgt_text, self.tokenizer.batch_decode(generated_utterances))

    @unittest.skip('broken')
    def test_loss_same_as_parlai_3B(self):
        config, input_ids, mask, batch_size = self.get_config_data()
        inputs_dict = {"input_ids": input_ids, "attention_mask": mask}
        src_text = [
            "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel "
            "like i'm going to throw up.\nand why is that?"
        ]
        tgt_text = ["I'm not sure, but I do know that social anxiety disorder is a mental disorder"]
        model_inputs = self.tokenizer(src_text, return_tensors='pt').to(torch_device)

        with torch.no_grad():
            output = self.model(**inputs_dict)[0]
        expected_shape = torch.Size((batch_size, input_ids.size(1), self.model.config.vocab_size))
        self.assertEqual(output.size(), expected_shape)
from .test_modeling_bart import _long_tensor, assert_tensors_close
class Blenderbot90MIntegrationTests(AbstractBlenderBotIntegrationTests):
    checkpoint_name = 'sshleifer/blenderbot-90M'
    tokenizer_cls = BlenderbotSmallTokenizer

    @slow
    def test_tokenization_same_as_parlai(self):
        tok = self.tokenizer
        self.assertListEqual(tok('sam'), [1,1384, 1])

    @slow
    def test_generation_same_as_parlai_90(self):
        src_text = [
            "Social anxiety\nWow, I am never shy. Do you have anxiety?\nYes. I end up sweating and blushing and feel like i'm going to throw up.\nand why is that?"
        ]
        tgt_text = ["I'm not sure, but I do know that social anxiety disorder is a mental disorder"]
        model_inputs = self.tokenizer(src_text, return_tensors='pt').to(torch_device)
        generated_utterances = self.model.generate(**model_inputs)
        self.assertListEqual(tgt_text, self.tokenizer.batch_decode(generated_utterances))

    @torch.no_grad()
    def test_samgen(self):
        input_ids = _long_tensor([[1384]])  # sam

        encoder_output = self.model.encoder(input_ids)[0]
        assert encoder_output.shape == (1,1,512)
        expected_slice = torch.tensor([0.0968, -0.0934, -0.1364, 0.0500, -0.0424, 0.1258, -0.0073, 0.0329, -0.1150, 0.0624])
        assert_tensors_close(encoder_output[0,0, :10], expected_slice, atol=1e-3)

        generated_utterances = self.model.generate(input_ids, min_length=20, max_length=30).tolist()
        expected_tokens = [1, 49, 15, 286, 474, 10, 1384, 5186, 20, 21, 8, 17,
                           50, 241, 1789, 6, 6299, 6, 9, 2147, 5, 2]
        self.assertListEqual(expected_tokens, generated_utterances)

    @torch.no_grad()
    def test_sam_forward(self):
        input_ids = _long_tensor([[1384]])  # sam
        ys = torch.tensor([[1, 49, 15, 286, 474, 10, 1384, 5186, 20, 21, 8, 17,
                           50, 241, 1789, 6, 6299, 6, 9, 2147, 5, 2]], dtype=torch.long)
        logits, *_ = self.model.forward(input_ids, decoder_input_ids=ys)
        #import ipdb; ipdb.set_trace()

        parlai = load_parlai()

        assert self.model.encoder.embed_tokens.weight[3, 3] == parlai.encoder.embeddings.weight[3,3]
        assert self.model.decoder.embed_tokens.weight[3, 3] == parlai.decoder.embeddings.weight[3, 3]

        self.assertEqual(num_parameters(self.model.encoder), 53613568)
        self.assertEqual(num_parameters(self.model.decoder), num_parameters(parlai.decoder))

        scores, preds, encoder_states = parlai.forward(input_ids, ys=ys[:,1:])
        enc_out, enc_mask = encoder_states
        assert self.model.encoder_states.shape == enc_out.shape
        assert_tensors_close(self.model.encoder_states[:,:,3], enc_out[:,:,3], atol=1e-3)
        desired_logits = torch.tensor([-0.1543, -4.0059, -1.1916, -4.3109, 3.9479, -0.6087, -0.0594], device=torch_device)
        assert_tensors_close(desired_logits, scores[0, 0, 3:10], atol=1e-4)
        assert_tensors_close(desired_logits, logits[0,0,3:10], atol=1e-4)

from durbango import *
def load_parlai():
    opt, dictionary = pickle_load('parlai_opt.pkl'), pickle_load('parlai_dict.pkl')
    parlai = TransformerGeneratorModel(opt, dictionary).eval().to(torch_device)
    state_dict = torch.load('bbot_state_dict.pt')
    parlai.load_state_dict(state_dict)
    return parlai
