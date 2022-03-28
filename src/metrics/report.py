
import copy
import os
from nbformat import write
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score
import matplotlib
import matplotlib.pyplot as plt

from data.annotations import Annotations
from data.helpers import Chat, Paragraph
from train.trainer import KerasModelWrapper, ModelWrapper

def fill_empty_seq_tokens(seq_tokens, oov_token=1, words=None, debug=True):
    """ Goes through the tokens created by keras tokenizer
        replaces the missing ones with unknown embedding
    """
    for idx in range(len(seq_tokens)):
        if not seq_tokens[idx]:
            seq_tokens[idx] = [oov_token]
        elif len(seq_tokens[idx]) > 1:
            if debug and words:
                print("Unknown seq token:", words[idx])
            seq_tokens[idx] = [oov_token]

class ResultsGenerator:
    """ Compute various metrics and generate pr-recall auc-roc figures """
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0
        self.true = []
        self.probas_pred = []

    def add_conf(self, true_hi:list, pred_hi:list):
        for true, pred in zip(true_hi, pred_hi):
            if true == 0 and pred == 0:
                self.tn += 1
            elif true == 0 and pred == 1:
                self.fp += 1
            elif true == 1 and pred == 0:
                self.fn += 1
            else:
                self.tp += 1

    def generate_results(self, write_dir, generate_plots, threshold):
        results = {}
        results['precision'] = self.tp / (self.tp+self.fp+1e-10)
        results['recall'] = self.tp / (self.tp+self.fn+1e-10)
        true, pred = np.array(self.true), np.array(self.probas_pred)
        results['roc-auc'] = roc_auc_score(true, pred)
        results['avg-prec-recall'] = average_precision_score(true, pred)
        if generate_plots:
            matplotlib.rcParams.update({'font.size': 16})
            # Roc curve
            fpr, tpr, roc_thresholds = roc_curve(true, pred)
            plt.plot(fpr, tpr, marker='.')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.subplots_adjust(bottom=0.15)
            save_path = os.path.join(write_dir, 'roc_curve.pdf')
            plt.savefig(save_path, format='pdf')

            # Precision Recall Curve
            plt.clf()
            precision, recall, thresholds = precision_recall_curve(true, pred)
            # Calculate point
            plt.plot(recall, precision, marker='.', label='logistic')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.subplots_adjust(bottom=0.15)
            save_path = os.path.join(write_dir, 'prec_recall.pdf')
            plt.savefig(save_path, format='pdf')

        print(results)
        return results

class Reporter:
    def __init__(self, config, model:ModelWrapper, annot: Annotations, wandb) -> None:
        self.config = config
        self.model = model
        self.annot = annot
        self.wandb = wandb
        self._results = ResultsGenerator()

class KerasReporter(Reporter):
    def __init__(self, config, model: KerasModelWrapper, annot, wandb) -> None:
        super().__init__(config, model, annot, wandb)

        self._chat_dicts = copy.deepcopy(annot._chat_dicts)

        # Fill chat dicts with the model predictions
        for file_idx in tqdm(self._chat_dicts.keys()):
            chat_dict = self._chat_dicts[file_idx]
            for para_dict in chat_dict[Chat.content]:
                words = para_dict[Paragraph.proc_tokens_rem]
                if not words:
                    para_dict[Paragraph.pred_hi] = []
                    continue
                seq_tokens = model.keras_tokenizer.texts_to_sequences(words)
                fill_empty_seq_tokens(seq_tokens, words=words, debug=True)
                pred = model.predict(seq_tokens)

                self._results.probas_pred.extend(pred.tolist())
                self._results.true.extend(para_dict[Paragraph.proc_tokens_hi])
                pred = pred > config['threshold']
                assert len(pred) == len(words)
                para_dict[Paragraph.pred_hi] = pred.astype('int').tolist()
                # Just look at the patient highlights
                if para_dict[Paragraph.person] == 'Patient':
                    self._results.add_conf(para_dict[Paragraph.proc_tokens_hi], pred)

    def report_results(self, ):
        write_dir = os.path.join(self.wandb.run.dir, 'test_annot')
        if self.config.get('save_documents', False):
            self.annot.report_results(self._chat_dicts,
                write_dir, self.config.get('report_debug', False))

        results = self._results.generate_results(self.wandb.run.dir,
            generate_plots=self.config.get('generate_plots', False),
            threshold=self.config['threshold'])
        self.wandb.log(results)

    def count_messages_per_chat(self, dir, file_name):
        self.annot.count_messages_per_chat(self._chat_dicts,
            WRITE_DIR=dir, file_name=file_name, only_patient=True)
