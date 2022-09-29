# coding=utf-8
# Modified from transformers (Hugging face)
#
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

try:
    import sklearn.metrics as metrics
    _has_sklearn = True
except (AttributeError, ImportError):
    _has_sklearn = False


def is_sklearn_available():
    return _has_sklearn

if _has_sklearn:
    def simple_accuracy(preds, labels):
        return (preds == labels).mean()

    def acc_and_f1_sk(preds,labels, start, end,name=''):
        active_labels=labels[:,start:end].flatten()
        active_preds=preds[:,start:end].flatten()

        true_labels=[l for (l,p) in zip(active_labels,active_preds) if l!=-100]
        true_preds = [p for (l, p) in zip(active_labels, active_preds) if l != -100]

        res_dict = metrics.classification_report(true_labels, true_preds, output_dict=True)
        acc = res_dict['accuracy']
        macro_f1 = res_dict['macro avg']['f1-score']
        macro_precision = res_dict['macro avg']['precision']
        macro_recall = res_dict['macro avg']['recall']
        res = {
            f"p{name}": macro_precision,
            f"r{name}": macro_recall,
            f"f1{name}": macro_f1,
            f"acc{name}": acc
        }
        return res


    def glue_compute_metrics(preds, labels,neg_idx=None):
        assert len(preds) == len(labels)

        res={}
        args=[
            {'preds':preds,'labels':labels, 'start':0, 'end':2127,'name':''},
            {'preds': preds, 'labels': labels, 'start':0, 'end':709, 'name': '_mlm'},
            {'preds': preds, 'labels': labels, 'start':709, 'end':1418, 'name': '_mim'},
            {'preds': preds, 'labels': labels, 'start':1418, 'end':2127, 'name': '_wpa'},
        ]

        for arg in args:
            r=acc_and_f1_sk(**arg)
            res.update(r)

        output_items=[
            'acc','acc_mlm','acc_mim','acc_wpa'
        ]
        res={k:res[k] for k in output_items}
        res['acc_avg']=(res['acc_mlm']+res['acc_mim']+res['acc_wpa'])/3.0
        # print(res,flush=True)
        return res


    def acc_and_f1_sk_dc(preds,labels,neg_idx=None):
        res_dict = metrics.classification_report(labels, preds, output_dict=True)
        acc = res_dict['accuracy']
        macro_f1 = res_dict['macro avg']['f1-score']
        macro_precision = res_dict['macro avg']['precision']
        macro_recall = res_dict['macro avg']['recall']
        res = {
            "p": macro_precision,
            "r": macro_recall,
            "f1": macro_f1,
            "acc": acc
        }
        return res


    def glue_compute_metrics_dc(preds, labels,neg_idx=None):
        assert len(preds) == len(labels)

        return acc_and_f1_sk_dc(preds, labels,neg_idx)