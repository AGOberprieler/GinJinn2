import argparse
import textwrap

class NewlineFormatter(argparse.ArgumentDefaultsHelpFormatter):
    '''NewlineFormatter

    "|R" can be used to introdue newlines.
    '''
    def _fill_text(self, text, width, indent):
        text = textwrap.dedent(text).strip()
        text = textwrap.indent(text, indent)
        text = text.split('|R')
        text = [textwrap.fill(line, width) for line in text]
        text = "\n".join(text)
        return text
