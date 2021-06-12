from text2text import Transformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class Translator(Transformer):

  def __init__(self, input_lines, pretrained_translator, src_lang='en', **kwargs):
    Transformer.__init__(self, input_lines, pretrained_translator, src_lang=src_lang, **kwargs)
    self.model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_translator)
    self.tokenizer = AutoTokenizer.from_pretrained(pretrained_translator)
    self.tokenizer.src_lang = src_lang

  def _translate(self, tgt_lang, **kwargs):
    tokenizer = self.tokenizer
    model = self.model
    src_lang = self.src_lang
    input_lines = self.input_lines

    if src_lang==tgt_lang:
      return self.input_lines
    if self.pretrained_translator==self.__class__.DEFAULT_TRANSLATOR:
      if tgt_lang not in self.__class__.LANGUAGES:
        raise ValueError(f'{tgt_lang} not found in {self.__class__.LANGUAGES}')
    encoded_inputs = tokenizer(input_lines, padding=True, return_tensors="pt")
    tgt_token_id = tokenizer.lang_code_to_id[tgt_lang]
    generated_tokens = model.generate(**encoded_inputs, forced_bos_token_id=tgt_token_id)
    return tokenizer.batch_decode(generated_tokens, skip_special_tokens=True) 

  def transform(self, **kwargs):
    return self._translate(**kwargs)