class Chat:
    id = 'chat_id'
    file_id = 'file_id'
    content = 'content'

class Paragraph:
    person = 'person'
    unproc_tokens = 'unprocessed_tokens'
    unproc_tokens_hi = 'unprocessed_tokens_highlights'
    proc_tokens = 'processed_tokens'
    proc_tokens_hi = 'processed_tokens_highlights'
    # Special characters and stopwords are removed, for training
    proc_tokens_rem = "processed_tokens_removed"
    pred_hi = "processed_tokens_pred_highlights"

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)