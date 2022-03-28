import pyonmttok
from nltk.corpus import stopwords
from data.helpers import Chat, Paragraph, has_numbers

stop_word_list = open('data/raw/stopword.txt', 'r').read().splitlines()

class Tokenizer:
    """ Custom Tokenizer that supports lossless detokenization """
    def __init__(self, remove_stopwords=False, remove_numbers=False):
        self._remove_numbers = remove_numbers
        self._stopwords = list(stopwords.words('english')) +  stop_word_list if remove_stopwords else []
        self._removed_token_start = '｟'
        self._removed_token_end = '｠'
        self._joiner = '￭'
        self._base_tokenizer = pyonmttok.Tokenizer("conservative", joiner_annotate=True)

    def _remove_tokens(self, tokens:list):
        token_list = []
        for org_token in tokens:
            token = org_token.lower()
            if (self._stopwords and token.strip() in self._stopwords) or \
                (self._remove_numbers and has_numbers(token)) or \
                ('http' in token or 'https' in token):
                token_list.append(f'{self._removed_token_start}{org_token}{self._removed_token_end}')
            else:
                token_list.append(org_token)
        return token_list

    def tokenize(self, unprocessed_tokens, unprocessed_tokens_highlights=None):
        """ Takes raw tokens and optionally highlights
        Returns processed tokens and its highlights, also full processed tokens for detokenization
        """
        removed_tokens = self._remove_tokens(unprocessed_tokens)
        sentence = " ".join(removed_tokens)
        processed_tokens_full, _ = self._base_tokenizer.tokenize(sentence)

        processed_tokens_removed = [token for token in processed_tokens_full
            if self._removed_token_start not in token and self._joiner not in token]
        processed_tokens_removed_highlights = self._map_highlights(unprocessed_tokens, unprocessed_tokens_highlights,
            processed_tokens_full, processed_tokens_removed)
        return processed_tokens_full, processed_tokens_removed, processed_tokens_removed_highlights

    def _map_highlights(self, unprocessed_tokens:list, unprocessed_tokens_highlights:list,
            processed_tokens_full:list, processed_tokens_removed:list):
        """ Create labels for the processed tokens
        """
        if not unprocessed_tokens_highlights: return None
        cur = 0
        processed_tokens_full_highlights = []
        # print(unprocessed_tokens)
        for token, highlight in zip(unprocessed_tokens, unprocessed_tokens_highlights):
            # Prefix
            while cur < len(processed_tokens_full) and processed_tokens_full[cur].startswith(self._joiner) and \
            processed_tokens_full[cur].endswith(self._joiner):
                last_highlight = processed_tokens_full_highlights[-1]
                processed_tokens_full_highlights += [last_highlight, last_highlight]
                cur += 2
            while cur < len(processed_tokens_full) and processed_tokens_full[cur].endswith(self._joiner) and \
                not processed_tokens_full[cur].startswith(self._joiner):
                processed_tokens_full_highlights.append(highlight)
                cur += 1
            # Actual word
            if processed_tokens_full[cur].startswith(self._removed_token_start):
                processed_tokens_full_highlights.append(0)
                cur += 1
            elif self._joiner not in processed_tokens_full[cur]:
                processed_tokens_full_highlights.append(highlight)
                cur += 1
            # Postfix
            while cur < len(processed_tokens_full) and (processed_tokens_full[cur].startswith(self._joiner) or \
                processed_tokens_full[cur].endswith(self._joiner)):
                while cur < len(processed_tokens_full) and processed_tokens_full[cur].startswith(self._joiner) and \
                processed_tokens_full[cur].endswith(self._joiner):
                    last_highlight = processed_tokens_full_highlights[-1]
                    processed_tokens_full_highlights += [last_highlight, last_highlight]
                    cur += 2
                while cur < len(processed_tokens_full) and processed_tokens_full[cur].startswith(self._joiner) and \
                    not processed_tokens_full[cur].endswith(self._joiner):
                    processed_tokens_full_highlights.append(highlight)
                    cur += 1
                while cur < len(processed_tokens_full) and processed_tokens_full[cur].endswith(self._joiner) and \
                    not processed_tokens_full[cur].startswith(self._joiner):
                    processed_tokens_full_highlights.append(highlight)
                    cur += 1

        assert len(processed_tokens_full_highlights) == len(processed_tokens_full)
        # Map highlights from processed full to processed removed
        processed_tokens_rem_highlights = []
        cur = 0
        for token in processed_tokens_removed:
            while cur < len(processed_tokens_full):
                if token == processed_tokens_full[cur].strip("|"):
                    processed_tokens_rem_highlights.append(processed_tokens_full_highlights[cur])
                    cur += 1
                    break
                cur += 1
        assert len(processed_tokens_rem_highlights) == len(processed_tokens_removed)
        return processed_tokens_rem_highlights

    def _map_preds(self, processed_tokens_full:list, proc_tokens_removed_pred_highlights:list,
        unprocessed_tokens:list):
        """ Map the predictions back to the unprocessed tokens """
        processed_tokens_full_pred_high = []
        # Map highlights from processed removed to processed full
        cur = 0
        for token in processed_tokens_full:
            if token.startswith(self._removed_token_start) or token.endswith(self._removed_token_end):
                processed_tokens_full_pred_high.append(0)
            elif token.startswith(self._joiner) and token.endswith(self._joiner):
                last_highlight = processed_tokens_full_pred_high[-1]
                processed_tokens_full_pred_high.append(last_highlight)
            elif token.startswith(self._joiner):
                last_highlight = processed_tokens_full_pred_high[-1]
                processed_tokens_full_pred_high.append(last_highlight)
            elif token.endswith(self._joiner):
                processed_tokens_full_pred_high.append(proc_tokens_removed_pred_highlights[cur])
            else:
                processed_tokens_full_pred_high.append(proc_tokens_removed_pred_highlights[cur])
                cur += 1
        assert len(processed_tokens_full_pred_high) == len(processed_tokens_full)
        # Map highlights from processed full to unprocessed
        unprocessed_tokens_pred_high = []
        cur = 0
        while cur < len(processed_tokens_full):
            token = processed_tokens_full[cur]
            highlight = processed_tokens_full_pred_high[cur]
            if token.startswith(self._joiner) and token.endswith(self._joiner):
                cur += 2
            elif self._joiner in token:
                cur += 1
            elif token.startswith(self._removed_token_start) or token.endswith(self._removed_token_end):
                unprocessed_tokens_pred_high.append(highlight)
                cur += 1
            else:
                unprocessed_tokens_pred_high.append(highlight)
                cur += 1
        assert len(unprocessed_tokens_pred_high) == len(unprocessed_tokens)
        return unprocessed_tokens_pred_high
