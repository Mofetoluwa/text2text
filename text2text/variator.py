from text2text import Transformer, Translator

class Variator(Translator):

  def transform(self, **kwargs):
    output_lines = []
    src_lang = self.src_lang
    original_input = self.input_lines

    for tgt_lang in self.__class__.LANGUAGES:
      Translator.__init__(self, original_input, self.pretrained_translator, src_lang=src_lang)
      translated = self._translate(tgt_lang=tgt_lang)
      Translator.__init__(self, translated, self.pretrained_translator, src_lang=tgt_lang)
      output_lines.append(self._translate(tgt_lang=src_lang))

    self.input_lines = original_input
    return output_lines 